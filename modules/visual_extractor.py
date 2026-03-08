import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel, AutoImageProcessor
import torch
import torch.nn as nn
from transformers import AutoModel


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_extractor = args.visual_extractor.lower()

        if self.visual_extractor in ['rad-dino', 'vit', 'dinov2', 'microsoft/rad-dino']:
            model_path = getattr(args, 'rad_dino_path')
            self.model_type = 'huggingface'
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Assume torchvision-like
            self.model_type = 'torchvision'
            self.pretrained = args.visual_extractor_pretrained
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images_or_pixel_values):
        if self.model_type == 'huggingface':
            # 输入是 pixel_values (B, 3, H, W) 经过 processor
            with torch.no_grad():
                outputs = self.model(**{"pixel_values": images_or_pixel_values})
            patch_embeddings = outputs.last_hidden_state  # (B, seq_len, hidden_size)
            cls_embeddings = outputs.pooler_output  # (B, hidden_size)
            patch_tokens = pool_patch_tokens_to_49(patch_embeddings)  # (B, 49, D)
            return patch_tokens, cls_embeddings
        else:
            # 输入是 torch tensor (B, 3, H, W)
            patch_feats = self.model(images_or_pixel_values)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            return patch_feats, avg_feats

# 你需要保证 pool_patch_tokens_to_49 支持 patch_embeddings 不同形状输入（如49、196、768等）


# class VisualExtractor(nn.Module):
#     def __init__(self, args):
#         super(VisualExtractor, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model_path = getattr(args, 'rad_dino_path')
#         self.model = AutoModel.from_pretrained(model_path).to(self.device)
#         self.model.eval()
#         for param in self.model.parameters():
#             param.requires_grad = False
#     def forward(self, image_processor):
#
#         with torch.no_grad():
#             outputs = self.model(**{"pixel_values": image_processor})
#         patch_embeddings = outputs.last_hidden_state  # (B, seq_len, hidden_size)
#         cls_embeddings = outputs.pooler_output  # (B, hidden_size)
#         patch_tokens = pool_patch_tokens_to_49(patch_embeddings)  # (B, 49, 768)
#         return patch_tokens, cls_embeddings





# class VisualExtractor(nn.Module):
#     def __init__(self, args):
#         super(VisualExtractor, self).__init__()
#         self.visual_extractor = args.visual_extractor
#         self.pretrained = args.visual_extractor_pretrained
#         model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
#         modules = list(model.children())[:-2]
#         self.model = nn.Sequential(*modules)
#         self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
#
#     def forward(self, images):
#         patch_feats = self.model(images)
#         avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
#         batch_size, feat_size, _, _ = patch_feats.shape
#         patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
#         return patch_feats, avg_feats
def pool_patch_tokens_to_49(patch_tokens):
    cls_token = patch_tokens[:, 0:1, :]          # (B, 1, 768)
    patch_only = patch_tokens[:, 1:, :]          # (B, 324, 768)

    x = patch_only.view(-1, 18, 18, 768).permute(0, 3, 1, 2)  # (B, 768, 18, 18)
    x = F.adaptive_avg_pool2d(x, output_size=(7, 7))         # (B, 768, 7, 7)
    x = x.permute(0, 2, 3, 1).reshape(-1, 49, 768)            # (B, 49, 768)

    return x