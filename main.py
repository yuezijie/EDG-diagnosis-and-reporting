import torch
from torch.utils.data import DataLoader,Subset
from models.MLLMEDRbackbone import Image2ReportModel
from EGDdatasets.TJdataset import EGDReportDataset, custom_collate_fn
from transformers import get_scheduler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
import os
import csv
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from options import MLLMEDROptions
import random

try:
    from nltk.corpus import wordnet
    _ = wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet')

options = MLLMEDROptions()
opts = options.parse()

def evaluate(model, val_loader, device, epoch=None, max_batches=None):
    model.eval()
    tokenizer = model.tokenizer
    smooth = SmoothingFunction().method4
    bleu1s, bleu2s, bleu3s, bleu4s = [], [], [], []
    meteors, rouges = [], []

    pred_list = []
    gt_list = []
    all_mucosa_preds, all_mucosa_trues = [], []
    all_disease_preds, all_disease_trues = [], []

    with torch.no_grad():
        for i, (imgs, reports, mucosa_labels, disease_labels, case_paths) in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            imgs = tuple(seq.to(device) for seq in imgs)
            mucosa_labels = mucosa_labels.to(device)
            disease_labels = disease_labels.to(device)

            preds, mucosa_logits, disease_logits = model.generate(imgs)
            targets = reports

            for pred, ref in zip(preds, targets):
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]
                bleu1s.append(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth))
                bleu2s.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
                bleu3s.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
                bleu4s.append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))
                meteors.append(meteor_score([ref.split()], pred.split()))
                rouges.append(rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(ref, pred)['rougeL'].fmeasure)
                pred_list.append(pred)
                gt_list.append(ref)

            all_mucosa_preds.extend(mucosa_logits.argmax(dim=1).cpu().tolist())
            all_mucosa_trues.extend(mucosa_labels.cpu().tolist())
            all_disease_preds.extend(torch.sigmoid(disease_logits).cpu().numpy())
            all_disease_trues.extend(disease_labels.cpu().numpy())

    if epoch is not None:
        os.makedirs(opts.save_dir, exist_ok=True) 
        csv_path = os.path.join(opts.save_dir, f"eval_epoch{epoch + 1}.csv")
        with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "pred_report", "gt_report"])
            for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
                writer.writerow([f"val_{i}", pred, gt])
        print(f"Saved prediction results to {csv_path}")

    mucosa_acc = accuracy_score(all_mucosa_trues, all_mucosa_preds)
    mucosa_precision, mucosa_recall, mucosa_f1, _ = precision_recall_fscore_support(all_mucosa_trues, all_mucosa_preds, average="macro", zero_division=0)

    disease_preds_bin = (torch.tensor(all_disease_preds) > 0.5).int().numpy()
    disease_precision, disease_recall, disease_f1, _ = precision_recall_fscore_support(all_disease_trues, disease_preds_bin, average="macro", zero_division=0)
    try:
        disease_auc = roc_auc_score(all_disease_trues, all_disease_preds, average="macro")
    except:
        disease_auc = 0.0

    return {
        "bleu1": sum(bleu1s) / len(bleu1s),
        "bleu2": sum(bleu2s) / len(bleu2s),
        "bleu3": sum(bleu3s) / len(bleu3s),
        "bleu4": sum(bleu4s) / len(bleu4s),
        "meteor": sum(meteors) / len(meteors),
        "rougeL": sum(rouges) / len(rouges),
        "mucosa_acc": mucosa_acc,
        "mucosa_recall": mucosa_recall,
        "mucosa_precision": mucosa_precision,
        "mucosa_f1": mucosa_f1,
        "disease_precision": disease_precision,
        "disease_recall": disease_recall,
        "disease_f1": disease_f1,
        "disease_auc": disease_auc
    }

def train(model, dataloader, optimizer, scheduler, device, epoch, num_epochs, max_steps=None):
    model.train()
    total_loss = 0
    tokenizer = model.tokenizer
    criterion_mucosa = torch.nn.CrossEntropyLoss()
    criterion_disease = torch.nn.BCEWithLogitsLoss()

    for step, (imgs, reports, mucosa_labels, disease_labels, case_paths) in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break

        imgs = tuple(seq.to(device) for seq in imgs)
        mucosa_labels = mucosa_labels.to(device)
        disease_labels = disease_labels.to(device)
        encoded = tokenizer(reports, return_tensors="pt", padding="longest", truncation=True, max_length=opts.tokenizer_max_length).to(device)

        loss, _, mucosa_logits, disease_logits = model((imgs, encoded.input_ids))
        mucosa_loss = criterion_mucosa(mucosa_logits, mucosa_labels)
        disease_loss = criterion_disease(disease_logits, disease_labels)

        total_task_loss = loss + opts.mucosa_loss_weight * mucosa_loss + opts.disease_loss_weight * disease_loss

        total_task_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += total_task_loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} | Step {step+1}/{len(dataloader)} | Loss: {total_task_loss.item():.4f}")
        torch.cuda.empty_cache()

    avg_loss = total_loss / (step + 1)
    print(f"Epoch {epoch+1} Training Loss (partial): {avg_loss:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = opts.data_path
    dataset = EGDReportDataset(root_dir,opts)
    if opts.train_test_ratio==1:
        n = len(dataset)
        indices = list(range(n))
        random.shuffle(indices)
        train_size = int(0.3 * n)
        train_indices = indices[:train_size]
        val_set = Subset(dataset, train_indices)
        train_set=dataset
    else:
        train_size = int(opts.train_test_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=opts.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    model = Image2ReportModel(opts).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.learning_rate)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                               num_training_steps=len(train_loader) * opts.num_epochs)

    best_bleu4 = 0
    for epoch in range(opts.num_epochs):
        train(model, train_loader, optimizer, scheduler, device, epoch, opts.num_epochs)
        metrics = evaluate(model, val_loader, device, epoch=epoch)
        print(f"Epoch {epoch+1} Validation Scores: {metrics}")

        if metrics["bleu4"] > best_bleu4:
            best_bleu4 = metrics["bleu4"]
            os.makedirs(opts.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(opts.save_dir,"best_model.pt"))
            print(f"Best model saved at epoch {epoch+1} with BLEU-4: {best_bleu4:.4f}")


if __name__ == "__main__":
    main()
