'''
Author: hxp
Introduction: 将不同单条染色体组合散布在背景图像上生成新的染色体图像
Time: 2022.01.21
'''
import os
import json
from webbrowser import Grail
import torch
import math
from PIL import Image

import cv2
from matplotlib.pyplot import axis
import numpy as np
from numpy.lib.ufunclike import fix
from numpy.random.mtrand import choice, random, sample
import random as rd

from scipy.fft import dst
import image_tool as imgTool
import generate_segmentation_dataset as gsd
import generate_background as gb
import imutils

singleSrcPath = "/home/guest01/projects/chromos/utils/chromotest/singleChromosome"
# jsonDir = "/home/guest01/projects/chromos/utils/chromotest/augmentation"
jsonDir = "/home/guest01/projects/chromos/dataset/segmentation_dataset/test_224seg_json"
backgroundDir = "/home/guest01/projects/chromos/utils/chromotest_result/augmentationSample"
savePath = "/home/guest01/projects/chromos/dataset/segmentation_dataset/train_overlap_fake_500smallImages_annotated"

# function: 根据json文件及单条染色体, 获取一一对应的放缩比例
# params: json文件路径, 单条染色体所在目录路径
# return: 放缩比例
def getChromosomeResize(jsonPath, srcPath):
    res = 0
    resize = {}
    jsonArea, _, _ = getJsonAreaAndPoints(jsonPath)
    srcArea = []
    _, srcDirs = imgTool.ReadPath(srcPath)
    for srcDir in srcDirs:
        srcDirImages, _ = imgTool.ReadPath(srcDir)
        for srcDirImage in srcDirImages:
            singleArea = getSingleChromosomeArea(srcDirImage)
            srcArea.append(singleArea)
    srcArea.sort(reverse=True)
    for i in range(min(len(jsonArea), len(srcArea))):
        sz = int(srcArea[i] / jsonArea[i])
        if sz in resize:
            resize[sz] += 1
            if resize[sz] > res:
                res = sz
        else:
            resize[sz] = 1
            if resize[sz] > res:
                res = sz
    # print(jsonArea)
    # print(srcArea)
    return res


# function: 根据每条染色体的极点，计算染色体与y轴的角度
# params: (x1, y1), (x2, y2)两点坐标
# return: 该染色体与y轴的角度
def getChromosomeAngle(point1, point2):
    p1I, p1J = point1
    p2I, p2J = point2
    dy = p2I - p1I
    dx = p2J - p1J
    # point1为minI点, point2为maxI点，计算线段与y轴的角度
    angle = math.atan2(dx, dy)
    angle = int(angle * 180 / math.pi)
    # print("angle: ", angle)
    return angle


# function: 根据json文件获取所有染色体实例的面积(像素点)
# params: json文件路径
# return: 所有染色体的 实例面积list，中心点list, 像素点list, 与y轴角度list
def getJsonAreaAndPoints(jsonPath):
    jsonArea = []
    centerList = []
    pointsList = []
    angleList = []
    data = json.load(open(jsonPath))
    jsonImage = gb.img_b64_to_arr(data.get("imageData"))
    if len(jsonImage.shape) == 3:
        jsonImage = cv2.cvtColor(jsonImage, cv2.COLOR_RGB2GRAY)
    instances, minI, minJ, maxI, maxJ = gb.getInstances(
        jsonImage.shape, data["shapes"])
    for instance in instances:
        tmpImage = np.zeros(jsonImage.shape, 'uint8')
        tmpImage[instance] = 255
        num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
            tmpImage, 8, ltype=cv2.CV_32S)
        if num_labels == 2:
            jsonArea.append(stats[1][4])
            center = centers[1].astype(int)
            points = np.argwhere(labels == 1)
            # print("points in getJsonAreaAndPoints: ", points, type(points))
            centerList.append(np.array([center[1], center[0]]))
            pointsList.append(points)
            angleList.append(getChromosomeAngle(points[0], points[len(points)-1]))
    jsonArea.sort(reverse=True)
    return jsonImage, jsonArea, centerList, pointsList, angleList, data["shapes"]


# function: 根据目录路径获取所有单条染色体面积
# params: 单条染色体图像路径
# return: 单条染色体的面积
def getSingleChromosomeArea(imagePath):
    image = cv2.imread(imagePath)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, tmpImage = cv2.threshold(image, 230, 255, 0)
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(tmpImage), 8, ltype=cv2.CV_32S)
    if num_labels == 2:
        return stats[1][4]
    return -1


# function: 根据单条染色体图像旋转角度并放缩提取出真实单条染色体像素坐标
# params: 单条染色体图像路径
# return: 单条染色体中心，单条染色体所有像素坐标
def getSingleChromosomePoints(imagePath, angle, rotateAngle):
    image = cv2.imread(imagePath)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # ang = rd.randint(0, 360)
    ang = angle + rotateAngle
    # 获取仿射变换矩阵
    M = cv2.getRotationMatrix2D((cX, cY), ang, 1.0)
    # 根据仿射变换矩阵执行仿射变换
    rotatedImage = cv2.warpAffine(image, M, (w, h), borderValue=255)
    # 对旋转后的图像进行放缩
    randWidth = rd.randint(110, 125)
    rotatedImage = imutils.resize(rotatedImage, width=randWidth)
    _, tmpImage = cv2.threshold(rotatedImage, 230, 255, 0)
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(tmpImage), 8, ltype=cv2.CV_32S)
    if num_labels == 2:
        # 通过argwhere查找数组中实例的所有像素坐标
        points = np.argwhere(labels == 1)
        center = centers[1].astype(int)
        return rotatedImage, np.array([center[1], center[0]]), points
    return rotatedImage, np.array([-1, -1]), np.array([[-1, -1]])


# function: 根据图像高宽及染色体掩膜，获取该染色体的轮廓坐标
# params: 掩膜，图像高，图像宽
# return: 掩膜轮廓坐标
def getContourPointsByMask(mask):
    image = np.zeros(mask.shape, 'uint8')
    image[mask] = 255
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return np.array(contours[0], dtype=float)

# function: 将染色体区域的像素点坐标采用bool数组表示
# params: 像素点坐标，图像高，图像宽
# return: (h, w)的bool数组
def pointsToMask(points, h, w):
    image = np.zeros((h, w), dtype=bool)
    for point in points:
        if point[0] < h and point[1] < w:
            image[point[0], point[1]] = True
    pos = np.where(image)
    hmin = np.min(pos[0])
    hmax = np.max(pos[0])
    wmin = np.min(pos[1])
    wmax = np.max(pos[1])
    # print("hwminmax", hmin, hmax, wmin, wmax)
    return image, hmin<hmax and wmin<wmax


# function: 在所有类的单条染色体中每类随机挑选两条
# params: 总共需要的染色体条数
# return: n条单条染色体的图像路径集合
def generateSingleChromosomeList(n):
    singleChromosomeList = []
    _, singleSrcDirs = imgTool.ReadPath(singleSrcPath)
    # 遍历每一类染色体选取其中一张单条染色体图像
    for singleSrcDir in singleSrcDirs:
        # 获取当前类目录中的所有图像路径
        singleSrcDirImages, _ = imgTool.ReadPath(singleSrcDir)
        index1 = rd.randint(0, len(singleSrcDirImages) - 1)
        index2 = rd.randint(0, len(singleSrcDirImages) - 1)
        singleChromosomeList.append(singleSrcDirImages[index1])
        singleChromosomeList.append(singleSrcDirImages[index2])
    while(len(singleChromosomeList) < n):
        chromosomeLabelIndex = rd.randint(0, len(singleSrcDirs) - 1)
        singleSrcDirImages, _ = imgTool.ReadPath(singleSrcDirs[chromosomeLabelIndex])
        singleChromosomeIndex = rd.randint(0, len(singleSrcDirImages) - 1)
        singleChromosomeList.append(singleSrcDirImages[singleChromosomeIndex])
    rd.shuffle(singleChromosomeList)
    return singleChromosomeList[:n]

# function: 在短小染色体类的单条染色体中每类随机挑选k条
# params: 总共需要的染色体条数
# return: n条单条染色体的图像路径集合
def generateSingleChromosomeSmallList(n):
    
    singleChromosomeList = []
    singleSrcDirs = ['/home/guest01/projects/chromos/utils/chromotest/singleChromosome/chromo16_solo',
                     '/home/guest01/projects/chromos/utils/chromotest/singleChromosome/chromo17_solo',
                     '/home/guest01/projects/chromos/utils/chromotest/singleChromosome/chromo18_solo',
                     '/home/guest01/projects/chromos/utils/chromotest/singleChromosome/chromo19_solo',
                     '/home/guest01/projects/chromos/utils/chromotest/singleChromosome/chromo20_solo',
                     '/home/guest01/projects/chromos/utils/chromotest/singleChromosome/chromo21_solo',
                     '/home/guest01/projects/chromos/utils/chromotest/singleChromosome/chromo22_solo',
                     '/home/guest01/projects/chromos/utils/chromotest/singleChromosome/chromoX_solo',
                     '/home/guest01/projects/chromos/utils/chromotest/singleChromosome/chromoY_solo']
    # 遍历每一类染色体选取其中一张单条染色体图像
    for i in range(len(singleSrcDirs)):
        singleSrcDirImages, _ = imgTool.ReadPath(singleSrcDirs[i])
        indexs = np.random.randint(len(singleSrcDirImages) - 1, size=6)
        for index in indexs:
            singleChromosomeList.append(singleSrcDirImages[index])
    while(len(singleChromosomeList) < n):
        labelDirIndex = rd.randint(len(singleSrcDirs)-1)
        singleSrcDirImages, _ = imgTool.ReadPath(singleSrcDirs[labelDirIndex])
        indexs = np.random.randint(len(singleSrcDirImages) - 1, size=6)
        for index in indexs:
            singleChromosomeList.append(singleSrcDirImages[index])
    rd.shuffle(singleChromosomeList)
    # print(singleChromosomeList)
    return singleChromosomeList[:n]

# function: 在所有的json和背景图像中随机选取进行组合
# params: json目录路径，背景图像目录路径
# return: 选取json文件路径，选取背景图像路径
def generateRandomJsonAndBackgroundPath(jsonPath, backgroundPath, i):
    jsonFiles, _ = imgTool.ReadPath(jsonPath)
    backgroundFiles, _ = imgTool.ReadPath(backgroundPath)
    # jsonIndex = rd.randint(0, len(jsonFiles) - 1)
    jsonIndex = 0
    # 用于生成不同种类的组合图像
    if i<150:
        jsonIndex = rd.randint(0, 76)
    elif i<300:
        jsonIndex = rd.randint(77, 263)
    else:
        jsonIndex = rd.randint(0, 263)
    backgroundIndex = rd.randint(0, len(backgroundFiles) - 1)
    return jsonFiles[jsonIndex], backgroundFiles[backgroundIndex]


# function: 根据json文件中染色体实例的中心点，将在每类染色体中随机选取的单条染色体置于该点处生成新的原始染色体图像
# params: json文件路劲，背景图像路径
# return: 自动生成的原始染色体图像, 该图像的所有染色体实例轮廓标注信息
def generateOriginChromosomeImage(jsonPath, backgroundPath):
    contoursPoints = []
    background = cv2.imread(backgroundPath)
    if len(background.shape) == 3:
        background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    # 获取json文件的所有染色体实例的中心和染色体像素坐标集
    _, _, jsonCenterList, _, angleList, _ = getJsonAreaAndPoints(jsonPath)
    # 获取随机的n条单挑染色体图像的路径
    singleChromosomeList = generateSingleChromosomeSmallList(len(jsonCenterList))
    print(jsonPath, len(jsonCenterList), len(singleChromosomeList))
    # 根据原始图像中每条染色体与y轴的角度全都随机旋转rotateAngle度
    rotateAngle = rd.randint(0, 360)
    for i in range(len(jsonCenterList)):
        single, singleCenter, singlePoints = getSingleChromosomePoints(
            singleChromosomeList[i], angleList[i], rotateAngle)
        singleMask, singleValid = pointsToMask(singlePoints, single.shape[0],
                                  single.shape[1])
        if singleValid == False:
            continue
        # 注意centers的i,j坐标是以(w,h)格式展示
        offsetI = jsonCenterList[i][0] - singleCenter[0]
        offsetJ = jsonCenterList[i][1] - singleCenter[1]
        backgroundPoints = singlePoints + [offsetI, offsetJ]
        backgroundMask, backgroundValid = pointsToMask(backgroundPoints, background.shape[0],
                                      background.shape[1])
        if backgroundValid == False:
            continue
        if background[backgroundMask].shape[0] == single[singleMask].shape[0]:
            background[backgroundMask] = single[singleMask]
            # 获取该条染色体的轮廓标注信息
            contour = getContourPointsByMask(backgroundMask)
            contourPoints = contour.reshape((contour.shape[0], 2))
            # contourPoints为array数组格式数据，需要转为list后才能用于构建labelme的json文件
            contoursPoints.append(contourPoints.tolist())
    return background, contoursPoints


# function: 根据json文件中染色体实例的中心点，在随机json中随机选取其中一条染色体置于该点处生成新的原始染色体图像
# params: json文件路劲，背景图像路径
# return: 自动生成的原始染色体图像, 该图像的所有染色体实例轮廓标注信息
def generateOriginChromosomeFixImage(jsonPath, backgroundPath, idx):
    contoursPoints = []
    background = cv2.imread(backgroundPath)
    if len(background.shape) == 3:
        background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    # 获取json文件的所有染色体实例的中心和染色体像素坐标集
    _, _, jsonCenterList, _, angleList, _ = getJsonAreaAndPoints(jsonPath)
    # 根据原始图像中每条染色体与y轴的角度全都随机旋转rotateAngle度
    for i in range(len(jsonCenterList)):
        singleFromJsonPath, _ = generateRandomJsonAndBackgroundPath(jsonDir, backgroundDir, idx)
        fromJsonImage, _, centerList, pointsList, _, shapesList = getJsonAreaAndPoints(singleFromJsonPath)
        chooseSingle = rd.randint(0, len(centerList)-1)
        singleCenter = centerList[chooseSingle]
        singlePoints = pointsList[chooseSingle]
        singleContourPoints = shapesList[chooseSingle]["points"]
        singleMask, _ = pointsToMask(singlePoints, fromJsonImage.shape[0],
                                  fromJsonImage.shape[1])
        # 注意centers的i,j坐标是以(w,h)格式展示
        offsetI = jsonCenterList[i][0] - singleCenter[0]
        offsetJ = jsonCenterList[i][1] - singleCenter[1]
        backgroundPoints = singlePoints + [offsetI, offsetJ]
        singleContour = (np.array(singleContourPoints) + [offsetI, offsetJ]).tolist()
        backgroundMask, _ = pointsToMask(backgroundPoints, background.shape[0],
                                      background.shape[1])
        # background[backgroundMask] = fromJsonImage[singleMask]
        background[singleMask] = fromJsonImage[singleMask]
        # 获取该条染色体的轮廓标注信息
        # contourPoints = contour.reshape((contour.shape[0], 2))
        # contourPoints为array数组格式数据，需要转为list后才能用于构建labelme的json文件
        # contoursPoints.append(contourPoints.tolist())
        contoursPoints.append(singleContourPoints)
    return background, contoursPoints

# function: 生成json中shapes字段内容
# params: 图像
# return: json（{"key": "val"}格式数据）
def generateOriginChromosomeJsonShapes(pointsList):
    # shapes字段包括label, points, group_id, shape_type, flag五个字段
    shapes = []
    label = "chromos"
    shape_type = "polygon"
    group_id = None
    flags = {}
    i = 1
    for points in pointsList:
        minx, miny = np.min(np.array(points), axis=0)
        maxx, maxy = np.max(np.array(points), axis=0)
        if len(points)>2 and minx<maxx and miny<maxy:
            annotatePoints = {
                "label": label,
                "points": points,
                "group_id": group_id,
                "shape_type": shape_type,
                "flags": flags
            }
            shapes.append(annotatePoints)
            i += 1
    # json.dump(shapes, open("test.json", 'w'))
    return shapes


# function: 生成json对应的标注实例label图像
# params: 图像保存路径，图像尺寸，json的shapes字段信息
# return: null
def generateOriginChromosomeJsonLabel(savePath, imageShape, shapes):
    label_name_to_value = {"_background_": 0}
    for i in range(len(shapes)):
        if len(shapes[i]["points"])>2:
            # label_name = 'chromos' + str(i+1)
            label_name = shapes[i]["label"]
            label_name_to_value[label_name] = i+1
    lbl, _ = gb.shapes_to_label(
        imageShape, shapes, label_name_to_value
    )
    gb.lblsave(savePath, lbl)

# function: 对生成的原始染色体图像生成labelme实例标注格式的json数据
# params: 图像路径, 存储输出路径，所有染色体实例的轮廓标注点
# return: json（{"key": "val"}格式数据）
def generateOriginChromosomeJson(imagePath, savePath, pointsList):
    image = cv2.imread(imagePath)
    version = "4.5.7"
    flags = {}
    shapes = generateOriginChromosomeJsonShapes(pointsList)
    imageData = gsd.transImage2RawData(imagePath)
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]
    annotateJson = {
        "version": version,
        "flags": flags,
        "shapes": shapes,
        "imagePath": imagePath,
        "imageData": imageData,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth
    }
    # 生成标注json文件
    filenamePre, _, _, _ = imgTool.parsePath(imagePath)
    jsonSavePath = savePath + "/" + filenamePre + ".json"
    json.dump(annotateJson, open(jsonSavePath, 'w'))
    # 生成mask的标注图像
    # lblSavePath = savePath + "/" + filenamePre + "_mask.png"
    # generateOriginChromosomeJsonLabel(lblSavePath, image.shape, shapes)


# function: 对生成的mask图像检测合法性(xmin<xmax, ymin<ymax)
# params: mask路径
# return: boxed, cnt
def siftMask(maskPath):
    mask = Image.open(maskPath)
    mask = np.array(mask)
    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]
    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]
    num_objs = len(obj_ids)
    boxes = []
    cnt = 0
    for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            if xmin>=xmax or ymin>=ymax:
                cnt += 1
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    return boxes, cnt


if __name__ == '__main__':
    # 根据染色体像素个数判断其放缩比例
    # resize = getChromosomeResize(jsonPath, srcPath)
    jsonSamples = [
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y1627.027.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y2140.049.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y2293.201.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y2295.031.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y2363.099.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y2672.071.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y2742.042.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y2752.149.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y3101.126.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y1622.001.O.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.110.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.111.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.115.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.155.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.042.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.046.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.047.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.049.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.050.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.080.A.json",
        "/home/guest01/projects/chromos/utils/chromotest/augmentation/21-Y342.105.A.json",
    ]
    for i in range(500):
        imagePath = savePath + "/fake" + str(i) + ".png"
        jsonPath, backgroundPath = generateRandomJsonAndBackgroundPath(jsonDir, backgroundDir, i)
        jsonPath = jsonSamples[rd.randint(0, len(jsonSamples)-1)]
        # jsonPath = "/home/guest01/projects/chromos/utils/chromotest/augmentation/18-Y2140.049.O.json"
        # jsonPath = "/home/guest01/projects/chromos/dataset/segmentation_dataset/origin_224images_annotated/21-Y342.001.A.json"
        if i%2==0:
            backgroundPath = "/home/guest01/projects/chromos/utils/chromotest_result/augmentationSample/white.png"
        image, contoursPoints = generateOriginChromosomeImage(jsonPath, backgroundPath)
        cv2.imwrite(imagePath, image)
        generateOriginChromosomeJson(imagePath, savePath, contoursPoints)
    
    # getChromosomeAngle(np.array([1,1+math.sqrt(3)]), np.array([2,1]))
    # t = "/home/guest01/projects/chromos/dataset/segmentation_dataset/chromosome_image_format/MasksFake500/"
    # ms, _ = imgTool.ReadPath(t)
    # for m in ms:
    #     tmpBox, tmpCnt = siftMask(m)
    #     if tmpCnt != 0:
    #         print(m, tmpBox, tmpCnt)