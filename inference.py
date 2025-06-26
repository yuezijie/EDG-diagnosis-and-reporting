import torch
from torch.utils.data import DataLoader
from models.MLLMEDRbackbone import Image2ReportModel
from EGDdatasets.TJdataset import EGDReportDataset, custom_collate_fn
from options import MLLMEDROptions
import os
import csv
from tqdm import tqdm

def inference(model, test_loader, device, save_path=None):
    model.eval()
    tokenizer = model.tokenizer
    all_preds = []
    all_gt = []
    all_case_paths = []

    with torch.no_grad():
        for imgs, reports, mucosa_labels, disease_labels, case_paths in tqdm(test_loader, desc="Running Inference"):
            imgs = tuple(seq.to(device) for seq in imgs)

            # 模型生成报告
            preds, _, _ = model.generate(imgs)

            all_preds.extend(preds)
            all_gt.extend(reports)
            all_case_paths.extend(case_paths)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_path", "generated_report", "ground_truth_report"])
            for path, pred, gt in zip(all_case_paths, all_preds, all_gt):
                writer.writerow([path, pred, gt])
        print(f"✅ Inference results saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options = MLLMEDROptions()
    opts = options.parse()

    # 加载测试集
    test_dataset = EGDReportDataset(opts.data_path, opts)
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 初始化模型并加载最佳权重
    model = Image2ReportModel(opts).to(device)
    best_model_path = os.path.join(opts.save_dir, "best_model.pt")
    assert os.path.exists(best_model_path), f"❌ Best model not found at {best_model_path}"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✅ Loaded best model from {best_model_path}")

    # 推理并保存
    save_csv = os.path.join(opts.save_dir, "test_predictions.csv")
    inference(model, test_loader, device, save_path=save_csv)

if __name__ == "__main__":
    main()
