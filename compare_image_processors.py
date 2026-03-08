import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
from transformers.image_processing_utils import BatchFeature

# 加载图像
img = Image.open("test.jpg").convert('RGB')

# AutoImageProcessor 处理
processor = AutoImageProcessor.from_pretrained("rad-dino")
base_model = AutoModel.from_pretrained("rad-dino", trust_remote_code=True)
proc_tensor = processor(images=img, return_tensors="pt")  # [3,256,256]
ans1=base_model(**proc_tensor)
# transforms.Compose 手动处理
manual_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5307]*3, std=[0.2583]*3)
])
manual_tensor = manual_transform(img).unsqueeze(0)  # [3,256,256]
inputs = {"pixel_values": manual_tensor}
ans2 = base_model(**inputs)


ans2 = base_model(manual_tensor)

manual_tensor