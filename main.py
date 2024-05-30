import cv2
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
from model import MaskDataset
from PIL import Image

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 实例化数据集
train_dataset = MaskDataset(image_dir='traindata/JPEGImages',
                            annotation_dir='traindata/Annotations',
                            imageset_dir='traindata/ImageSets',
                            subset='train',
                            transform=transform)

# 加载训练好的模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 更新为使用权重
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 假设只有两类，有口罩和无口罩
model.load_state_dict(torch.load('mask_classifier.pth'))
model.eval()

# OpenCV的人脸检测器
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    # 从摄像头读取图像
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测图像中的人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 对于检测到的每个人脸
    for (x, y, w, h) in faces:
        # 绘制矩形框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 获取人脸区域
        face = frame[y:y + h, x:x + w]
        # 将人脸区域转换为PIL图像
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        # 将人脸区域转换为模型需要的格式
        face = transform(face).unsqueeze(0)

        # 使用模型进行预测
        with torch.no_grad():
            output = model(face)
            _, predicted = torch.max(output, 1)

        # 根据预测结果在人脸框上显示标签
        label = 'Mask' if predicted.item() == 1 else 'No Mask'
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 显示结果图像
    cv2.imshow('Mask Detection', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
