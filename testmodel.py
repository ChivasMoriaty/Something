import os
import cv2
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练好的模型
model = torchvision.models.resnet18(weights=None)  # 使用None作为权重参数
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 假设只有两类，有口罩和无口罩
model.load_state_dict(torch.load('mask_classifier.pth', map_location=torch.device('cpu')))
model.eval()


# 读取测试集图像列表
def read_test_images_list(imageset_dir):
    with open(os.path.join(imageset_dir, 'test.txt'), 'r') as f:
        lines = f.read().splitlines()
    return [line.split()[0].split('\\')[-1] for line in lines]


# 测试模型
def test_model(model, image_dir, imageset_dir):
    image_list = read_test_images_list(imageset_dir)
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        label = 'Mask' if predicted.item() == 1 else 'No Mask'

        print(f'Image: {image_name} - Prediction: {label}')


# 调用测试函数
test_model(model, 'traindata/JPEGImages', 'traindata/ImageSets')
