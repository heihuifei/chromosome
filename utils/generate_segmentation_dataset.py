'''
Author: hxp
Introduction: 对染色体原图进行规则切割，并采用传统边缘检测方法提取不同类别的单条染色体生成数据集
Time: 2021.11.13
'''
import cv2
from pylab import *
from json import dumps
import json
from base64 import b64encode

import image_tool as imgTool


# function: 生成labelme实例标注格式的json数据
# params: 图像路径, 存储输出路径
# return: json（{"key": "val"}格式数据）
def generateJson(imagePath, savePath):
    version = "4.5.7"
    flags = {}
    shapes = generateJsonShapes(imagePath)
    imageData = transImage2RawData(imagePath)
    imageHeight = 896
    imageWidth = 1017
    annotateJson = {
        "version": version,
        "flags": flags,
        "shapes": shapes,
        "imagePath": imagePath,
        "imageData": imageData,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth
    }
    filenamePre, _, _, _ = imgTool.parsePath(imagePath)
    jsonSavePath = savePath + "/" + filenamePre + ".json"
    json.dump(annotateJson, open(jsonSavePath, 'w'))


# function: 生成json中shapes字段内容
# params: 图像路径
# return: json（{"key": "val"}格式数据）
def generateJsonShapes(imagePath):
    # shapes字段包括label, points, group_id, shape_type, flag五个字段
    shapes = []
    label = "chromos"
    shape_type = "polygon"
    group_id = None
    flags = {}
    pointsList = getPointsByContour(imagePath)
    for points in pointsList:
        print(points)
        annotatePoints = {
            "label": label,
            "points": points,
            "group_id": group_id,
            "shape_type": shape_type,
            "flags": flags
        }
        shapes.append(annotatePoints)
    # json.dump(shapes, open("test.json", 'w'))
    return shapes


# function: 将图像进行编码解码后转为utf-8字符串格式
# params: 图像路径
# return: string
def transImage2RawData(imagePath):
    with open(imagePath, "rb") as f:
        # 采用字节流格式读取图片信息
        byteImage = f.read()
    base64Image = b64encode(byteImage)
    utf8Image = base64Image.decode("utf-8")
    return utf8Image


# function: 根据图像采用Canny边缘算法获取所有的标注坐标点列表
# params: 图像路径
# return: [[[x, y],[x, y],[x, y]], [[x, y],[x, y]]] (轮廓坐标list)
def getPointsByContour(imagePath):
    img = cv2.imread(imagePath)  # 读取图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR图像转换为GRAY图像
    low_thre = 100
    high_thre = 230
    cnt_contours = 0
    max_contours = 0
    best_thre = 100
    while low_thre <= high_thre:
        ret, thresh = cv2.threshold(gray, low_thre, 255,
                                    cv2.THRESH_BINARY)  #循环动态获取最好的二值图像
        canny = cv2.Canny(thresh, 0, 255)  # 通过canny算子获取该效果较好的二值图像的轮廓
        contours, hierarchy = cv2.findContours(
            canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )  # 绘制轮廓,第三个参数是轮廓的索引（在绘制单个轮廓时有用。要绘制所有轮廓，请传递-1）
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) >= 80 and cv2.contourArea(
                    contours[i]) <= 6000:
                cnt_contours += 1
        if cnt_contours > max_contours:
            max_contours = cnt_contours
            best_thre = low_thre
        low_thre += 1
        cnt_contours = 0
    chromos = []
    ret, thresh = cv2.threshold(gray, best_thre, 255,
                                cv2.THRESH_BINARY)  #循环动态获取最好的二值图像
    canny = cv2.Canny(thresh, 0, 255)  # 通过canny算子获取该效果较好的二值图像的轮廓
    contours, hierarchy = cv2.findContours(
        canny, cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)  # 绘制轮廓,第三个参数是轮廓的索引（在绘制单个轮廓时有用。要绘制所有轮廓，请传递-1）
    point0 = []
    for x in range(0, contours[0].shape[0]):
        point0.append(
            [float(contours[0][x][0][0]),
             float(contours[0][x][0][1])])
    chromos.append(point0)
    for i in range(1, len(contours)):
        if cv2.contourArea(contours[i]) >= 80 and cv2.contourArea(
                contours[i]) <= 6000:
            if abs(
                    cv2.contourArea(contours[i]) -
                    cv2.contourArea(contours[i - 1])) >= (
                        cv2.contourArea(contours[i]) / 10):
                point = []
                for x in range(0, contours[i].shape[0]):
                    point.append([
                        float(contours[i][x][0][0]),
                        float(contours[i][x][0][1])
                    ])
                chromos.append(point)
    return chromos


# function: 根据图像采用连通域获取所有的标注坐标点列表
# params: 图像路径
# return: [[[x, y],[x, y],[x, y]], [[x, y],[x, y]]] (轮廓坐标list)
def getPointsByConnect(imagePath, thre):
    image = cv2.imread(imagePath)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 源图片, 阈值, 填充色, 阈值类型，根据将小于thre的像素点置0(黑色)或置填充色
    # type=0(小于thre置0, 大于阈值置填充色)，type=1与0相反，type=3(小于thre置0, 大于阈值保持原色)type=4与3相反
    _, threshImage = cv2.threshold(grayImage, thre, 255, 0)
    cv2.imwrite("/home/guest01/projects/chromos/utils/chromotest/labelme/thre.png", threshImage)
    chromos = []
    return chromos

if __name__ == '__main__':
    srcPath = "/home/guest01/projects/chromos/utils/chromotest/segmentation"
    dstPath = "/home/guest01/projects/chromos/utils/chromotest/labelme"
    imagePaths, _ = imgTool.ReadPath(srcPath)
    for imagePath in imagePaths:
        # generateJson(imagePath, dstPath)
        getPointsByConnect(imagePath, 200)
