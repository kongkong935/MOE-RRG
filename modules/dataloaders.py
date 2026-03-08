import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import torch
from torch.nn.utils.rnn import pad_sequence




#imp_mask处理函数
def make_imp_mask(impression_batch, null_token='[NIMP]'):
    """
    impression_batch: Tuple[str] 或 List[str]
    null_token: 标记为空白的特殊字符串（默认 '[NIMP]'）
    返回: LongTensor (B,)
    """
    mask = [0 if (t.strip() == null_token) else 1 for t in impression_batch]
    return torch.tensor(mask, dtype=torch.long)

#视角映射
VIEW2ID = {"PA": 0, "AP": 1, "LATERAL": 2, "LL": 3}

#就诊次序分桶
def bucket_visit(v):
    if v == 1:       return 0
    elif v == 2:     return 1
    elif 3 <= v <= 5: return 2
    elif 6 <= v <= 10: return 3
    else:            return 4

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer,bert_tokenizer, split, shuffle):
        self.args = args
        self.tokenizer=tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.split = split
        if self.args.visual_extractor == "resnet101":
            if split == 'train':
                # 训练集的变换，包含随机裁剪和水平翻转，用于数据增强
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),  # 图像均值
                                         (0.229, 0.224, 0.225))  # 图像标准差
                ])
            else:
                # 验证集和测试集的变换，固定大小裁剪，无数据增强
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),  # 图像均值
                                         (0.229, 0.224, 0.225))  # 图像标准差
                ])
        elif self.args.visual_extractor == "rad-dino":
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5307] * 3, std=[0.2583] * 3)
            ])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split,self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split,self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'pin_memory': True
        }
        super().__init__(**self.init_kwargs)



    def collate_fn(self,data):
        """
        将 Dataset 返回的一批样本组织为批次张量，供模型训练或推理使用。
        每条数据包含：图像、报告、视角、病史描述等字段。
        """
        # ------------- 按列拆包 -------------
        (image_processor_batch, report_batch,report_mask_batch,viewposition_batch,indication_batch,comparison_batch,
         visitorder_batch,prior_report_batch,impression_batch,seq_length_batch) = zip(*data)

        # # ------------- 图像打包 -------------
        image_processor_batch = torch.stack(image_processor_batch, dim=0)  # (B, 3, H, W)

        #-------------- 报告对齐 ------------
        max_seq_length = max(seq_length_batch)
        target_batch = np.zeros((len(report_batch), max_seq_length), dtype=int)
        target_masks_batch = np.zeros((len(report_batch), max_seq_length), dtype=int)
        for i, report_ids in enumerate(report_batch):
                target_batch[i, :len(report_ids)] = report_ids
        for i, report_masks in enumerate(report_mask_batch):
                target_masks_batch[i, :len(report_masks)] = report_masks

        # ==========  MoE 路由特征（视角 / 比较 / 就诊）==========
        viewposition_MOE = torch.LongTensor([VIEW2ID.get(v.upper(), 4) for v in viewposition_batch])  # (B,)
        visitorder_MOE   =  torch.LongTensor([bucket_visit(v) for v in visitorder_batch])  # (B,)


        # ----------- 需要编码的辅助信息 先拼接然后编码(BERT tokenizer 会自动 pad 到 batch 内最长) -------------
        aux_text_batch = []
        for ind_text, pri_text, comp_flag in zip(indication_batch, prior_report_batch, comparison_batch):
            parts = [f"[INDICATION] {ind_text.strip()}"]
            if int(comp_flag) == 1 and pri_text.strip() != "":
                prior_text = pri_text.strip()
            else:
                prior_text = "[NHPR]"
            parts.append(f"[PRIOR] {prior_text}")
            aux_text_batch.append(" ".join(parts))

        aux_text_tokenizer    = self.bert_tokenizer(list(aux_text_batch), padding=True, truncation=True,max_length=128,return_tensors='pt')
        imp_tokenizer         = self.bert_tokenizer(list(impression_batch),padding=True, truncation=True,max_length=128,return_tensors='pt')
        imp_mask              = make_imp_mask(impression_batch)
        # ------------- 返回 -------------
        return {
            # ------------- 图像打包 -------------
            "image_processor": image_processor_batch,  # (B, H, W)

            # -------------- 报告对齐 ------------
            "report": torch.LongTensor(target_batch),  # (B, Lr)
            "report_mask": torch.FloatTensor(target_masks_batch),  # (B, Lr)

            # ========== MoE 路由特征（视角 / 比较 / 就诊）==========
            "viewposition_MOE": viewposition_MOE,  # (B,)
            "visitorder_MOE": visitorder_MOE,  # (B,)

            #==========  辅助文本（BERT）     ==========
            "aux_text_batch": aux_text_tokenizer,
            "impression": imp_tokenizer,
            "imp_mask": imp_mask

        }




    #
    # @staticmethod
    # def collate_fn(data):
    #     """
    #     对齐 Dataset 返回的 9 个字段。
    #     """
    #     (image_id_batch, images_batch, report_batch, viewpositions, indications, comparisons,
    #      visitorders,prior_reports, impressions) = zip(*data)
    #
    #     # === 图像打包 ===
    #     image_batch = torch.stack(images_batch, dim=0)  # (Batch_size, 3, H, W)
    #
    #
    #
    #     # === 返回一个 dict 更方便使用 ===
    #     return {
    #
    #     }





    # @staticmethod
    # def collate_fn(data):
    #     image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch = zip(*data)
    #     image_batch = torch.stack(image_batch, 0)
    #     max_seq_length = max(seq_lengths_batch)
    #
    #     target_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
    #     target_masks_batch = np.zeros((len(report_ids_batch), max_seq_length), dtype=int)
    #
    #     for i, report_ids in enumerate(report_ids_batch):
    #         target_batch[i, :len(report_ids)] = report_ids
    #
    #     for i, report_masks in enumerate(report_masks_batch):
    #         target_masks_batch[i, :len(report_masks)] = report_masks
    #
    #     return image_id_batch, image_batch, torch.LongTensor(target_batch), torch.FloatTensor(target_masks_batch)