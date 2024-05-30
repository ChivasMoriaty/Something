import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import xml.etree.ElementTree as ET
from PIL import Image


# 从ImageSets文件中读取图像列表
def read_image_lists(imageset_dir, subset):
    with open(os.path.join(imageset_dir, subset + '.txt'), 'r') as f:
        lines = f.read().splitlines()
    image_names = [line.split()[0].split('\\')[-1].replace('.png', '') for line in lines]
    return image_names


# 自定义数据集类
class MaskDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, imageset_dir, subset, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images = read_image_lists(imageset_dir, subset)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name + '.png')
        annotation_path = os.path.join(self.annotation_dir, img_name + '.xml')
        image = Image.open(img_path).convert('RGB')
        # 解析XML文件获取标签
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        # 获取第一个目标的标签（有口罩或无口罩）
        label = root.find('object/name').text
        label = 1 if label == 'with_mask' else 0
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 实例化数据集
train_dataset = MaskDataset(image_dir='traindata/JPEGImages',
                            annotation_dir='traindata/Annotations',
                            imageset_dir='traindata/ImageSets',
                            subset='train',
                            transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义模型
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 假设只有两类，有口罩和无口罩

# 检查是否存在权重文件
weight_file = 'mask_classifier.pth'
if os.path.exists(weight_file):
    # 加载权重文件
    model.load_state_dict(torch.load(weight_file))
    print("Loaded pretrained weights.")
else:
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), weight_file)
    print("Trained model saved.")

# 此时可以使用 model 进行预测
