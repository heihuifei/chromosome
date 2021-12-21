import os
import types
import numpy as np
import torch
import sys
import torchvision
import cv2
import random
import time
import datetime

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

sys.path.append("../")
from detection.engine import train_one_epoch, evaluate
import detection.utils as utils
import detection.transforms as T

sys.path.append("../../")
import utils.image_tool as imgTool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# function: 随机生成一个rgb颜色
# params: null
# return: rgb值
def randomColor():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return (b, g, r)


# function: 将图片转为tensor格式且数据为float类型
# params: 图片路径
# return: 图片tensor
def toTensor(imagePath):
    image = cv2.imread(imagePath)
    # 使用cv2.imread读取出来的image为numpy.ndarray类型
    assert type(
        image) == np.ndarray, 'the img type is {}, but ndarry expected'.format(
            type(image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将数组array转换为张量tensor
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    # 255也可以改为256
    return image.float().div(255)


# function: 使用输出的模型对单张图像进行预测返回预测结果
# params: 单张图像路径，实例类别数
# return: 图像预测结果（box, masks, scores, labels）
def predictImage(imagePath, numClasses):
    image = toTensor(imagePath)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False, num_classes=numClasses)
    model = model.to(device)
    # 测试时需要使用eval
    model.eval()
    weight = torch.load(
        "../../outputModels/segmentationModels/maskrcnn_resnet50_for29_modelweights.pth"
    )
    model.load_state_dict(weight)
    # 需要添加[]使得其为一张三通道的图像否则会报错
    # 需要将其输入至device保证输入图像和参数类型都为cpu或者cuda
    prediction = model([image.to(device)])
    return prediction


# function: 对单张图像的预测结果可视化展示
# params: 单张图像路径，单张图像预测结果
# return: null
def showPrediction(imagePath, prediction):
    image = cv2.imread(imagePath)
    maskImage = image.copy()
    boxImage = image.copy()
    # showImage = np.zeros(image.shape, 'uint8')
    # prediction是gpu带grad数据，需要对齐进行detach去梯度以及转cpu操作后才能转换为numpy格式
    scores = prediction[0]['scores'].cpu().detach().numpy()
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    # (instanceNum, 1, height, weight)内部值很小需要进行mul(255).byte()转为整数值
    masks = prediction[0]['masks'].mul(255).byte().cpu().detach().numpy()
    for i in range(masks.shape[0]):
        if scores[i] >= 0.8:
            color = randomColor()
            maskBool = masks[i][0] != 0
            maskImage[maskBool] = color
            cv2.rectangle(boxImage, (int(boxes[i][0]), int(boxes[i][1])),
                          (int(boxes[i][2]), int(boxes[i][3])),
                          color,
                          thickness=1)
            cv2.putText(boxImage,
                        text="chromos",
                        org=(int(boxes[i][0]), int(boxes[i][1] + 10)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                        color=color)
    cv2.imwrite("/home/guest01/projects/chromos/utils/result.png",
                cv2.addWeighted(boxImage, 0.7, maskImage, 0.3, 0))


# function: 对单张图像的预测实例逐条分割
# params: 单张图像路径，单张图像预测结果，分割结果存储路径
# return: null
def segmentInstance(imagePath, prediction, savePath):
    image = cv2.imread(imagePath)
    filename, _, _, _ = imgTool.parsePath(imagePath)
    saveDir = savePath + "/" + filename + "/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    scores = prediction[0]['scores'].cpu().detach().numpy()
    masks = prediction[0]['masks'].mul(255).byte().cpu().detach().numpy()
    for i in range(masks.shape[0]):
        if scores[i] >= 0.8:
            cpImage = np.zeros(image.shape, dtype=np.uint8)
            maskBool = masks[i][0] != 0
            cpImage[maskBool] = image[maskBool]
            cv2.imwrite(saveDir + filename + "_" + str(i) + ".png", cpImage)


if __name__ == "__main__":
    # toTensor("/home/guest01/projects/chromos/utils/161938.044.A.JPG")
    pre = predictImage("/home/guest01/projects/chromos/utils/161938.044.A.JPG",
                       numClasses=2)
    # showPrediction("/home/guest01/projects/chromos/utils/161938.044.A.JPG",
    #                pre)
    segmentInstance(
        "/home/guest01/projects/chromos/utils/161938.044.A.JPG", pre,
        "/home/guest01/projects/chromos/segmentation/segmentInstances")
