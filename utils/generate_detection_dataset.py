'''
Author: hxp
Introduction: 根据分割的标注数据集, 生成图像的
Time: 2021.11.13
'''
import queue
import os

import cv2
from mim import download
from pylab import *
from json import dumps
import json
from base64 import b64encode

from sympy import re

import image_tool as imgTool
import matplotlib.pyplot as plt
import generate_background as gb
import generate_classification_dataset as gcd
import generate_segmentation_dataset as gsd

maskDir = "/home/guest01/projects/chromos/dataset/segmentation_dataset/train_origin77and187images_fake1500_withClear_annotated/"
saveDir = "/home/guest01/projects/chromos/dataset/detection_dataset/chromos/chromos_origin_fake1500/train/"


# function: 根据标注txt文件在图像上绘制旋转的四边形
# params: 输入图像, 输入标注文件
# return: null
def drawRotatedRectangle(imagePath, annPath):
    image = cv2.imread(imagePath)
    with open(annPath, 'r') as f:
        annlines = f.readlines()
    # 将标注文件中的imagesource: xx, gsd: xx忽视
    annPoints = annlines[2:]
    for annPoint in annPoints:
        points = annPoint.split()
        # print(type(annPoint))
        x1, y1, x2, y2, x3, y3, x4, y4, _, _ = points
        print(x1, y1)
        cv2.line(image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=3)
        cv2.line(image, pt1=(int(x2), int(y2)), pt2=(int(x3), int(y3)), color=(255, 0, 0), thickness=3)
        cv2.line(image, pt1=(int(x3), int(y3)), pt2=(int(x4), int(y4)), color=(255, 0, 0), thickness=3)
        cv2.line(image, pt1=(int(x4), int(y4)), pt2=(int(x1), int(y1)), color=(255, 0, 0), thickness=3)
    cv2.imwrite("result.png", image)


# function: 根据输入四个点的坐标数组，返回按照顺时针排列的四个点坐标数组
# params: 坐标点数组
# return: 坐标点list
def generateClockwisePoints(points):
    # sort by points[:, 0], then by poins[:, 1], return index, 先按w排再按h排
    sort1 = np.lexsort((points[:,1], points[:,0]))
    # 根据第一列、第二列优先级进行排序后获取左上顶点
    left_top = points[sort1[0]]
    # 根据第一列、第二列优先级进行排序后获取右下顶点
    right_down = points[sort1[3]]
    # sort by points[:, 1], then by poins[:, 0], return index, 先按h排再按w排
    sort2 = np.lexsort((points[:,0], points[:,1]))
    top_right = points[sort2[0]]
    # 如果h相等, 则取w更大的点
    if points[sort2[0]][1]==points[sort2[1]][1]:
        top_right = points[sort2[1]]
    down_left = points[sort2[3]]
    # 如果h相等, 则取w更小的点
    if points[sort2[2]][1]==points[sort2[3]][1]:
        down_left = points[sort2[2]]
    return np.array([left_top, top_right, right_down, down_left])


# function: 根据json文件路径生成其对应图像的所有对象的box坐标点列表
# params: json文件路径
# return: json对应图像的所有对象的box坐标点列表
def generateImageAnnPoints(jsonPath):
    annBoxsPoints = []
    data = json.load(open(jsonPath))
    jsonImage = gb.img_b64_to_arr(data.get("imageData"))
    if len(jsonImage.shape) == 3:
        jsonImage = cv2.cvtColor(jsonImage, cv2.COLOR_RGB2GRAY)
    instances, minI, minJ, maxI, maxJ = gb.getInstances(
        jsonImage.shape, data["shapes"])
    # 获取每个对象的点坐标
    for i in range(len(instances)):
        tmpImage = np.zeros(jsonImage.shape, 'uint8')
        tmpImage[instances[i]] = 255
        # 通过np.where找到该实例的所有点坐标
        areaPoints = np.column_stack(np.where(tmpImage > 0))
        # 将坐标x, y互换
        areaPoints = areaPoints[:, ::-1]
        min_rect = cv2.minAreaRect(areaPoints)
        # 获取到矩形框的四个顶点坐标(int值), 具有随机性, 坐标(w, h)=(x, y)
        box = np.int0(cv2.boxPoints(min_rect))
        annBoxPoints = generateClockwisePoints(box)
        annBoxsPoints.append(annBoxPoints)
    return annBoxsPoints


# function: 根据json文件路径生成每张图像的rotated标注文件
# params: json文件路径
# return: null
def generateImageAnnfile(jsonPath, savePath):
    filepre, _, _, _ = imgTool.parsePath(jsonPath)
    data = json.load(open(jsonPath))
    jsonImage = gb.img_b64_to_arr(data.get("imageData"))
    annImage = jsonImage
    # 获取图像所有对象的点坐标
    annBoxsPoints = generateImageAnnPoints(jsonPath)
    # 在图像上绘制其标注box并保存
    for annBoxPoints in annBoxsPoints:
        annImage = cv2.drawContours(annImage, [annBoxPoints], 0, [0, 0, 0], 1)
    cv2.imwrite(savePath + filepre + ".png", annImage)
    # 根据图像名生成其标注文件
    if os.path.exists(savePath + filepre + ".txt"):
        os.remove(savePath + filepre + ".txt")
    with open(savePath + filepre + ".txt", 'a+') as f:
        f.writelines('imagesource:GoogleEarth')
        f.write('\n')
        f.writelines('gsd:0.12')
        f.write('\n')
        for annBoxPoints in annBoxsPoints:
            boxAnnLine = str(annBoxPoints[0][0]) + ' ' + str(annBoxPoints[0][1]) + ' ' \
                + str(annBoxPoints[1][0]) + ' ' + str(annBoxPoints[1][1]) + ' ' \
                + str(annBoxPoints[2][0]) + ' ' + str(annBoxPoints[2][1]) + ' ' \
                + str(annBoxPoints[3][0]) + ' ' + str(annBoxPoints[3][1]) + ' ' + 'chromos 0'
            f.writelines(boxAnnLine)
            f.write('\n')



if __name__ == '__main__':
    # 根据染色体像素个数判断其放缩比例
    # drawRotatedRectangle("/home/guest01/projects/chromos/utils/P0003.png", "/home/guest01/projects/chromos/utils/P0003.txt")
    maskPaths, _ = imgTool.ReadPath(maskDir)
    for maskPath in maskPaths:
        print(maskPath)
        _, filepost, _, _ = imgTool.parsePath(maskPath)
        if filepost == ".json":
            generateImageAnnfile(maskPath, saveDir)
    # generateImageAnnfile("/home/guest01/projects/chromos/dataset/segmentation_dataset/origin_100images_annotated/18-Y2090.130.O.json")
