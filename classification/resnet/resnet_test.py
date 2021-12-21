'''
Author: hxp
Introduction: 对训练好的resnet模型进行预测验证
Time: 2021-11-13
'''
import json
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.append("../../")
import utils.image_tool as imgTool

transform = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 进行分类时类别名并非按照1~22顺序排列，而是按照10, 11, 12...排列
with open('./class_names.json', 'r') as f:
    class_names = json.load(f)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# function: 验证模型的准确率，预测多张图像并验证统计
# params: 验证图像路径
# return: 所有类别图像的预测准确率
def ValidModel(imagesPath):
    acc = 0
    wrong = 0
    imagesAcc = []
    for i in range(1, 23):
        chromoi_acc = 0
        chromoi_wrong = 0
        chromoi_imgs, _ = imgTool.ReadPath(imagesPath + "chromo" + str(i) +
                                         "_solo")
        for now_img in chromoi_imgs:
            now_img_pre = prediectImg(now_img)
            if now_img_pre == "chromo" + str(i) + "_solo":
                acc = acc + 1
                chromoi_acc = chromoi_acc + 1
                print("这是一个预判正确的图片!")
            else:
                wrong = wrong + 1
                chromoi_wrong = chromoi_wrong + 1
                print("这是一个预判错误的图片!", "染色体号: ", i, "图片路径: ", now_img)
        imagesAcc.append(chromoi_acc / (chromoi_acc + chromoi_wrong))
    print("正确数: ", acc, "错误数: ", wrong, "总数: ", acc + wrong)
    return imagesAcc, acc / (acc + wrong)


# function: 使用输出的模型对单张图像进行预测并输出预测类别
# params: 单张图像路径
# return: 图像预测结果（类别名）
def prediectImg(imagePath):
    net = torch.load(
        "../../outputModels/classficationModels/resnet18_for25_modelall.pth")
    net = net.to(device)
    torch.no_grad()
    img = Image.open(imagePath)
    img = transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    print('this picture maybe :', class_names[predicted[0]])
    return class_names[predicted[0]]


if __name__ == '__main__':
    imagesAcc, all_acc = ValidModel(
        "../../dataset/classification_dataset/valid/")
    print("this is all labels acc: ", imagesAcc)
    print("this is all_acc: ", all_acc)
