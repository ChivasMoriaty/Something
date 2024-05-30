import os
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练好的模型
model = torchvision.models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('mask_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# OpenCV的人脸检测器
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


# 读取测试集图像列表
def read_test_images_list(imageset_dir):
    with open(os.path.join(imageset_dir, 'test.txt'), 'r') as f:
        lines = f.read().splitlines()
    return [line.split()[0].split('\\')[-1] for line in lines]


# 显示图像并进行预测
def display_prediction(image_list, image_dir):
    index = 0
    while True:
        image_name = image_list[index]
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = image[y:y + h, x:x + w]
            face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face = transform(face).unsqueeze(0)

            with torch.no_grad():
                output = model(face)
                _, predicted = torch.max(output, 1)

            label = 'Mask' if predicted.item() == 1 else 'No Mask'
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Mask Detection', image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('a'):
            index = max(0, index - 1)
        elif key == ord('d'):
            index = min(len(image_list) - 1, index + 1)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


# 获取图像列表并调用显示函数
image_list = read_test_images_list('traindata/ImageSets')
display_prediction(image_list, 'traindata/JPEGImages')
