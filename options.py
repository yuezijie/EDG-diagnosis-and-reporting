from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in



class MLLMEDROptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="/VisCom-SSD-2/yzj/datasets/集成数据-2025")

        self.parser.add_argument("--save_dir",
                                 type=str,
                                 help="save_path",
                                 default= "checkpoints")


        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=16)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=5e-5)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=100)

        self.parser.add_argument("--mucosa_loss_weight",
                                 type=float,
                                 help="loss weight of the mucosa classification",
                                 default=0.3)
        self.parser.add_argument("--disease_loss_weight",
                                 type=float,
                                 help="loss weight of the disease classification",
                                 default=0.3)

        self.parser.add_argument("--tokenizer_max_length",
                                 type=int,
                                 help="tokenizer_max_length",
                                 default=128)

        self.parser.add_argument("--img_num",
                                 type=int,
                                 help="max img num for each patient",
                                 default=16)

        self.parser.add_argument("--tokens_per_frame",
                                 type=int,
                                 help="tokens_per_frame",
                                 default=8)

        self.parser.add_argument("--record_name",
                                 type=str,
                                 help="record_excel_name",
                                 default="newrecord.xlsx")

        self.parser.add_argument("--report_colomn",
                                 type=str,
                                 help="report_colomn_of_record",
                                 default="Findings_En")

        self.parser.add_argument("--name_colomn",
                                 type=str,
                                 help="name_colomn_of_record",
                                 default="姓名")

        self.parser.add_argument("--vision_model_path",
                                 type=str,
                                 help="path_of_vision_model",
                                 default="/VisCom-SSD-2/yzj/paper9/MLLM-EDR/CLIP")

        self.parser.add_argument("--decoder_path",
                                 type=str,
                                 help="path_of_decoder",
                                 default="/VisCom-SSD-2/yzj/paper9/MLLM-EDR/pre/opt-2.7b/")

        self.parser.add_argument("--prompt",
                                 type=str,
                                 help="prompt_instruction",
                                 default="Please generate an esophagogastroduodenoscopy report based on the provided visual features. Assistant:")

        self.parser.add_argument("--mucosa_classes",
                                 nargs="+",
                                 type=str,
                                 help="List of mucosa class names",
                                 default=[
                                     "慢性非萎缩性胃炎", "慢性萎缩性胃炎(C1)", "慢性萎缩性胃炎(C2)",
                                     "慢性萎缩性胃炎(C3)", "慢性萎缩性胃炎(O1)",
                                     "活动性胃炎"
                                 ])
        self.parser.add_argument("--num_mucosa_classes",
                                 type=int,
                                 help="num_mucosa_classes",
                                 default=6)

        self.parser.add_argument("--num_disease_classes",
                                 type=int,
                                 help="num_disease_classes",
                                 default=19)

        self.parser.add_argument("--disease_classes",
                                 nargs="+",
                                 type=str,
                                 help="List of disease class names",
                                 default=[
                                     "十二指肠球部溃疡(A1)", "十二指肠球部溃疡(A2)", "十二指肠球部溃疡(S1)",
                                     "十二指肠球部溃疡(S2)",
                                     "反流性食管炎(LA-A)", "反流性食管炎(LA-B)", "反流性食管炎(LA-C)",
                                     "糜烂", "胃底息肉", "胃体息肉", "胃窦息肉",
                                     "胃角溃疡(A1)", "胃角溃疡(A2)", "胃窦溃疡(A1)", "胃窦溃疡(A2)",
                                     "隆起糜烂", "早期胃癌", "进展期胃癌", "早期食管癌"
                                 ])
        self.parser.add_argument("--knowledge_path",
                                 type=str,
                                 help="knowledge_path",
                                 default="knowledge.xlsx")

        self.parser.add_argument("--train_test_ratio",
                                 type=float,
                                 help="train_test_split_ratio",
                                 default=0.7)



    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
