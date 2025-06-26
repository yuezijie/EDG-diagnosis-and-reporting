import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer, CLIPTokenizer
import pandas as pd
from .modules import CrossAttentionBlock

class Image2ReportModel(nn.Module):
    def __init__(self,opts):
        super().__init__()

        self.num_frames = opts.img_num
        self.tokens_per_frame = opts.tokens_per_frame
        self.prompt = opts.prompt

        self.clip = CLIPModel.from_pretrained(opts.vision_model_path)
        self.processor = CLIPProcessor.from_pretrained(opts.vision_model_path)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(opts.vision_model_path)
        self.vision_width = self.clip.vision_model.config.hidden_size
        self.text_width = self.clip.text_model.config.hidden_size

        self.decoder = AutoModelForCausalLM.from_pretrained(opts.decoder_path)
        self.tokenizer = AutoTokenizer.from_pretrained(opts.decoder_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.visual_proj = nn.Linear(self.vision_width, self.decoder.config.hidden_size)
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.vision_width, nhead=8, batch_first=True),
            num_layers=1
        )

        self.knowledge_proj = nn.Linear(self.text_width, self.vision_width)

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(self.vision_width, nhead=8, context_dim=self.vision_width) for _ in range(3)
        ])

        self.mucosa_classifier = nn.Linear(self.vision_width, opts.num_mucosa_classes)
        self.disease_classifier = nn.Linear(self.vision_width, opts.num_disease_classes)

        for param in self.clip.text_model.parameters():
            param.requires_grad = False

        self.clip = self.clip.eval()

        df = pd.read_excel(opts.knowledge_path)
        self.knowledge_texts = df.iloc[:, 0].tolist()
        with torch.no_grad():
            inputs = self.clip_tokenizer(self.knowledge_texts, padding=True, return_tensors="pt")
            knowledge_feats = self.clip.text_model(**inputs).last_hidden_state[:, 0, :]
            self.knowledge_embeds = self.knowledge_proj(knowledge_feats)

    def encode_images(self, images):
        batch_embeds = []
        batch_cls_tokens = []
        device = images[0][0].device

        for seq in images:
            token_seq = []
            cls_seq = []
            for img in seq:
                from torchvision.transforms.functional import to_pil_image
                inputs = self.processor(images=to_pil_image(img.cpu()), return_tensors="pt", do_rescale=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    vision_out = self.clip.vision_model(**inputs).last_hidden_state
                    vision_out = vision_out.squeeze(0)

                    cls_token = vision_out[0:1, :]
                    patch_tokens = vision_out[1:, :].transpose(0, 1).unsqueeze(0)
                    pooled_patches = F.adaptive_avg_pool1d(patch_tokens, self.tokens_per_frame).squeeze(0).transpose(0, 1)
                    tokens = torch.cat([cls_token, pooled_patches], dim=0)
                    token_seq.append(tokens)
                    cls_seq.append(cls_token)

            token_tensor = torch.cat(token_seq, dim=0)
            pooled_tokens = F.adaptive_avg_pool1d(token_tensor.transpose(0, 1).unsqueeze(0), self.num_frames * (self.tokens_per_frame + 1)).squeeze(0).transpose(0, 1)
            batch_embeds.append(pooled_tokens)
            cls_tensor = torch.cat(cls_seq, dim=0)
            batch_cls_tokens.append(torch.mean(cls_tensor, dim=0))

        batch_embeds = torch.stack(batch_embeds).to(device)
        batch_cls_tokens = torch.stack(batch_cls_tokens).to(device)
        batch_embeds = self.self_attn(batch_embeds)

        return batch_embeds, batch_cls_tokens

    def get_prompt_token_embed(self, device, batch_size):
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(device)
        prompt_embeds = self.decoder.get_input_embeddings()(inputs.input_ids)
        return prompt_embeds.expand(batch_size, -1, -1)

    def forward(self, inputs):
        imgs, target_input_ids = inputs
        visual_embeds, cls_tokens = self.encode_images(imgs)

        mucosa_logits = self.mucosa_classifier(cls_tokens)

        B = visual_embeds.size(0)
        device = visual_embeds.device
        knowledge_cls = self.knowledge_embeds.to(device).unsqueeze(0).expand(B, -1, -1)

        for block in self.cross_blocks:
            visual_embeds = block(visual_embeds, knowledge_cls)

        visual_embeds_proj = self.visual_proj(visual_embeds)
        prompt_embeds = self.get_prompt_token_embed(device, B)
        prefix_embeds = torch.cat([prompt_embeds, visual_embeds_proj], dim=1)

        patch_tokens = visual_embeds[:, 1:, :]
        patch_feats = patch_tokens.mean(dim=1)
        disease_logits = self.disease_classifier(patch_feats)

        if target_input_ids is None:
            generated_text = self.generate(imgs)
            return generated_text, mucosa_logits, disease_logits

        prefix_len = prefix_embeds.size(1)
        target_embeds = self.decoder.get_input_embeddings()(target_input_ids)
        inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)

        label_pad = torch.full((B, prefix_len), -100, dtype=torch.long, device=device)
        labels = torch.cat([label_pad, target_input_ids], dim=1)

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True
        )

        return outputs.loss, outputs.logits, mucosa_logits, disease_logits

    def generate(self, imgs, max_len=128):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            visual_embeds, cls_tokens = self.encode_images(imgs)

            B = visual_embeds.size(0)
            knowledge_cls = self.knowledge_embeds.to(device).unsqueeze(0).expand(B, -1, -1)
            for block in self.cross_blocks:
                visual_embeds = block(visual_embeds, knowledge_cls)

            visual_embeds_proj = self.visual_proj(visual_embeds)
            prompt_embeds = self.get_prompt_token_embed(device, B)
            prefix_embeds = torch.cat([prompt_embeds, visual_embeds_proj], dim=1)

            generated = torch.full((B, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=device)

            for _ in range(max_len):
                token_embeds = self.decoder.get_input_embeddings()(generated)
                inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
                outputs = self.decoder(inputs_embeds=inputs_embeds)
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tokens], dim=1)
                if (next_tokens == self.tokenizer.eos_token_id).all():
                    break

            generated_text = [self.tokenizer.decode(g[1:], skip_special_tokens=True) for g in generated]

            patch_tokens = visual_embeds[:, 1:, :]
            patch_feats = patch_tokens.mean(dim=1)
            disease_logits = self.disease_classifier(patch_feats)
            mucosa_logits = self.mucosa_classifier(cls_tokens)

            return generated_text, mucosa_logits, disease_logits




if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import ToTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_image = ToTensor()(Image.new("RGB", (224, 224), color=(255, 255, 255)))
    imgs = [[dummy_image.to(device) for _ in range(8)] for _ in range(2)]

    target_texts = [
        "Esophagus: mild congestion and erosion observed.",
        "Gastric mucosa generally healthy, minor polyps present."
    ]

    model = Image2ReportModel().to(device)

    encoded = model.tokenizer(
        target_texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=64
    ).to(device)

    loss, logits, mucosa_logits, disease_logits = model((imgs, encoded.input_ids))

    print("Report loss:", loss.item())
    print("Mucosa logits:", mucosa_logits)
    print("Disease logits:", disease_logits)

    results, mucosa_pred, disease_pred = model.generate(imgs)
    print("Generated Reports:")
    for r in results:
        print(r)