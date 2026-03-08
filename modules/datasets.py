import json
import os
from transformers import AutoModel, AutoImageProcessor
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer



class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split ,transform):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.tokenizer=tokenizer
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.transform=transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['reports'] = self.tokenizer(self.examples[i]['reports'])
            self.examples[i]['mask'] = [1] * len(self.examples[i]['reports'])
    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        # ==== 1. 获取当前样本 ====
        example = self.examples[idx]

        # ==== 2. 报告编码信息 ====
        report           = example['reports']                    # token ID 序列（如 [101, 2034, ...]）
        report_mask      = example['mask']                       # tokenID 掩码
        seq_length       = len(report)                           # report长度，方便后续对齐

        # ==== 3. MoE变量   ====
        viewposition     = example['viewposition']       # 视角：PA / AP / LATERAL / LL
        visitorder       = example['visitorder']         # 第几次就诊（1=首诊）

        # ==== 4. Prompt变量 ====
        comparison       = example['comparison']         # 是否比较（0/1）
        prior_report     = example['prior_report']       # 上一次报告文本；若无则为 "[NHPR]"
        indication       = example['indication_pure']    # INDICATION 段落（干净文本）
        impression       = example['impression']         # IMPRESSION 段文本（可能为空）


        # ==== 5. 图像加载 ====
        image_path = example['image_path']               # 相对路径：p10/p100.../s50.../xxx.jpg
        full_path  = os.path.join(self.image_dir, image_path)
        image = Image.open(full_path).convert('RGB')     # 加载图像为 RGB 模式
        image_processor=self.transform(image) # [3,256,256]
        #image_id = full_path                             # 可作为唯一标识符


        # ==== 6. 打包返回 ====
        sample = (
            image_processor, # 图像预处理张量
            report,          # 报告 token 序列
            report_mask,
            viewposition,    # 视角类别
            indication,      # 病人主诉（用于 prompt）
            comparison,      # 有无比较（0/1）
            visitorder,      # 第几次随访
            prior_report,    # 上一次报告文本
            impression,      # 当前报告的 impression（用于生成多段落）
            seq_length       # 方便后续对齐report和report_mask
        )

        return sample

