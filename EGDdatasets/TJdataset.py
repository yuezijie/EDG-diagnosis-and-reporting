import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from transformers import AutoTokenizer
import random
from collections import Counter


class EGDReportDataset(Dataset):
    def __init__(self, root_dir,
                 opts,
                 img_exts=(".png", ".jpg", ".jpeg")):

        self.excel_name = opts.record_name
        self.id_col = opts.name_colomn
        self.report_col = opts.report_colomn

        self.tokenizer = AutoTokenizer.from_pretrained(opts.decoder_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = opts.tokenizer_max_length

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        self.mucosa_classes = opts.mucosa_classes
        self.disease_classes = opts.disease_classes

        self.disease_class_to_idx = {cls: idx for idx, cls in enumerate(self.disease_classes)}
        self.disease_counter = Counter()
        self.samples = []
        self.img_num=opts.img_num

        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.isdir(class_path):
                continue

            mucosa_label = None
            for mucosa in self.mucosa_classes:
                if mucosa in class_dir:
                    mucosa_label = self.mucosa_classes.index(mucosa)
                    break
            if mucosa_label is None:
                continue

            disease_labels = []
            for token in class_dir.split("_"):
                if token not in self.mucosa_classes and token in self.disease_class_to_idx:
                    disease_labels.append(token)
                    self.disease_counter[token] += 1

            xlsx_path = os.path.join(class_path, self.excel_name)
            if not os.path.exists(xlsx_path):
                continue

            df = pd.read_excel(xlsx_path, dtype=str)
            if self.id_col not in df.columns or opts.report_colomn not in df.columns:
                continue

            for case_id, report in zip(df[self.id_col], df[opts.report_colomn]):
                case_dir = os.path.join(class_path, case_id)
                if not os.path.isdir(case_dir):
                    continue

                img_paths = [
                    os.path.join(case_dir, fn)
                    for fn in sorted(os.listdir(case_dir))
                    if fn.lower().endswith(img_exts)
                ]
                if not img_paths:
                    continue

                self.samples.append({
                    "images": img_paths,
                    "report": report,
                    "mucosa": mucosa_label,
                    "disease_tokens": disease_labels,
                    "case_path": case_dir 
                })

        pd.DataFrame({
            "Disease": list(self.disease_classes),
            "Count": [self.disease_counter.get(k, 0) for k in self.disease_classes]
        }).to_csv("disease_stats.csv", index=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_paths = sample["images"]
        if len(img_paths) >self.img_num:
            img_paths = random.sample(img_paths, self.img_num)
        imgs = [self.transform(Image.open(p).convert("RGB")) for p in img_paths]
        imgs = torch.stack(imgs, dim=0)

        report = sample["report"]
        mucosa_label = sample["mucosa"]

        disease_vec = torch.zeros(len(self.disease_classes))
        for token in sample["disease_tokens"]:
            idx = self.disease_class_to_idx.get(token)
            if idx is not None:
                disease_vec[idx] = 1.0

        return imgs, report, mucosa_label, disease_vec, sample["case_path"]


def custom_collate_fn(batch):
    imgs_batch, report_batch, mucosa_batch, disease_batch, case_paths = zip(*batch)
    return list(imgs_batch), list(report_batch), torch.tensor(mucosa_batch), torch.stack(disease_batch), list(case_paths)


if __name__ == "__main__":
    ROOT = "/VisCom-SSD-2/yzj/datasets/集成数据-2025"
    dataset = EGDReportDataset(ROOT)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size
    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    imgs, reports, mucosa_labels, disease_labels, case_paths = next(iter(train_loader))

    print("图像 shape:", imgs[0].shape)
    print("示例报告:", reports[1])
    print("黏膜标签:", mucosa_labels[1])
    print("疾病标签:", disease_labels[1])
    print("病例目录路径:", case_paths[1])
