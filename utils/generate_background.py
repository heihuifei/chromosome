'''
Author: hxp
Introduction: 对染色体原图进行规则切割，并采用传统边缘检测方法提取不同类别的单条染色体生成数据集
Time: 2021.11.13
'''
import math
import uuid
import json
import io
import os
import base64
import os.path as osp

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.lib.ufunclike import fix
from numpy.random.mtrand import choice, random, sample
import pandas as pd
import random as rd

from scipy.fft import dst

import image_tool as imgTool
import generate_segmentation_dataset as gsd

label_name_to_value = {"_background_": 0, "chromos": 1}
handPadding = [70, 35]


# function: 将labelme中的shapes字段坐标点数组转为mask(源于labelme源码)
# params: 图像尺寸，坐标点...
# return: mask坐标
def shape_to_mask(img_shape,
                  points,
                  shape_type=None,
                  line_width=10,
                  point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px)**2 + (cy - py)**2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


# function: 将labelme中的shapes字段坐标点数组转为label标记(源于labelme源码)
# params: 图像尺寸，shapes字段内容，{"background": 0}
# return: 图像矩阵
def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id
    return cls, ins


# function: 将labelme中根据shapes生成的图像矩阵存储为图像(源于labelme源码)
# params: 图像名称
# return: null
def lblsave(filename, lbl):
    import imgviz

    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError("[%s] Cannot save the pixel-wise class label as PNG. "
                         "Please consider using the .npy format." % filename)

# function: 用于将imageData转为image
# params: 图像的imageData数据
# return: img_pil
def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil


# function: 将imageData的数据转为array格式的image
# params: 图像的b64编码并被解码后的数据
# return: image
def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr

# function: 将imageData的b64编码数据转为array格式的image
# params: 图像的b64编码数据
# return: image
def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


# function: 根据图像像素值分布，按概率分布随机生成cnt个(r,g,b)值
# params: 图像，生成(r,g,b)个数
# return: [[r1,g1,b1] [r2,g2,b2]...] (type=array)
def generateRandRgbs(image, cnt):
    # 将image数组变型为的rgb像素值数组
    cpImage = image[handPadding[0]:(image.shape[0]-handPadding[0]), handPadding[1]:(image.shape[1]-handPadding[1]), :]
    imageRgb = cpImage.reshape(cpImage.shape[0] * cpImage.shape[1], 3)
    # 使用pandas库统计list中不同(r,g,b)出现频率,可以通过索引查找元素
    imageRgbMap = pd.value_counts(imageRgb.tolist())
    # 将Series先转为list后再转为array便于后续直接一次访问多个下标数据
    imageRgbKeys = np.array(list(imageRgbMap.index))
    imageRgbVals = np.array(list(imageRgbMap.values))
    # 生成rgb种类数的目标索引数组[0,1,2,3,...len(imageRgbKeys)]
    randI = np.arange(len(imageRgbKeys))
    # 按照(r,g,b)出现频率构造其随机生成的概率数组
    randP = np.array(np.divide(imageRgbVals, imageRgbVals.sum()))
    # 从randI中随机选取元素，size表示随机选取个数，replace表示是否可以选取重复(可放回)，p表示概率
    randIndex = np.random.choice(randI,
                                 size=cnt,
                                 replace=True,
                                 p=randP.ravel())
    # print("随机rgb值: ", imageRgbKeys[randIndex])
    return imageRgbKeys[randIndex]


# function: 根据shapes字段内容获取图像所有实例的mask的bool数组及其分布边界位置
# params: 图像尺寸，shapes字段内容
# return: [mask0, mask1...] (type=array), minI, minJ, maxI, maxJ
def getInstances(img_shape, shapes):
    minI = []
    minJ = []
    maxI = []
    maxJ = []
    instances = []
    for shape in shapes:
        points = shape["points"]
        shape_type = shape.get("shape_type", None)
        mask = shape_to_mask(img_shape[:2], points, shape_type)
        instances.append(mask)
        # points中[x, y]是按照[长, 宽]，和image读取的(宽,长)不一样
        pointsArr = np.array(points)
        # 获取points中所有列(axis=0)的最值并添加到minI/minJ/maxI/maxJ中
        minI.append(np.min(pointsArr, axis=0)[1])
        minJ.append(np.min(pointsArr, axis=0)[0])
        maxI.append(np.max(pointsArr, axis=0)[1])
        maxJ.append(np.max(pointsArr, axis=0)[0])
    return instances, int(np.min(np.array(minI))), int(np.min(np.array(minJ))),int(np.max(np.array(maxI))),int(np.max(np.array(maxJ)))


# function: 根据所有染色体分布的边界minI,minJ,maxI,maxJ选取图像无染色体分布的四个角所有像素点
# params: 图像，染色体分布上边界，左边界，下边界，右边界，返回的三维矩阵h，返回的三维矩阵w
# return: [[[r1,g1,b1], [r2,g2,b2]...]]
def cutImageNonChromos(image, minI, minJ, maxI, maxJ, cnt):
    # LU，LD，RD，RU分别代表左上角，左下角，右下角，右上角四个角子图
    fourImageSegs = []
    fourImageSegs.append(image[handPadding[0]:minI, handPadding[1]:(image.shape[1]-handPadding[1])])
    fourImageSegs.append(image[handPadding[0]:(image.shape[0]-handPadding[0]), handPadding[1]:minJ])
    fourImageSegs.append(image[maxI:(image.shape[0]-handPadding[0]), handPadding[1]:(image.shape[1] - handPadding[1])])
    fourImageSegs.append(image[handPadding[0]:image.shape[0]-handPadding[0], maxJ:(image.shape[1] - handPadding[1])])
    fillPoints = fourImageSegs[0].reshape(-1)
    fillPoints = np.concatenate((fillPoints, fourImageSegs[1].reshape(-1)), axis=0)
    fillPoints = np.concatenate((fillPoints, fourImageSegs[2].reshape(-1)), axis=0)
    fillPoints = np.concatenate((fillPoints, fourImageSegs[3].reshape(-1)), axis=0)
    np.random.shuffle(fillPoints)
    # fillPoints = np.concatenate((fillPoints, choosePoints), axis=0)
    # np.random.shuffle(fillPoints)
    return fillPoints[:cnt]


# function: 根据非染色体区域对染色体区域进行覆盖填充
# params: 图像，染色体分布上边界，左边界，下边界，右边界
# return: 图像，待填充上边界，左边界，下边界，右边界
def fillImageNonChromos(cpImage, minI, minJ, maxI, maxJ):
    image = cpImage
    # 选取填充区域的iU, iD, jL, jR
    randLoc = [[handPadding[0],minI,handPadding[1], image.shape[1]-handPadding[1]], 
                       [handPadding[0],image.shape[0]-handPadding[0], handPadding[1],minJ], 
                       [maxI,image.shape[0]-handPadding[0], handPadding[1], image.shape[1]-handPadding[1]], 
                       [handPadding[0],image.shape[0]-handPadding[0], maxJ, image.shape[1]-handPadding[1]]
    ]
    dire = rd.randint(0,3)
    if dire == 0 or dire == 2:
        offsetJ = rd.randint(handPadding[1]-minJ, image.shape[1]-handPadding[1]-maxJ)
        if (maxI-minI) > (randLoc[dire][1]-randLoc[dire][0]): # 区域高度不够填充
            cpImage[minI:(minI+randLoc[dire][1]-randLoc[dire][0]), minJ:maxJ] = image[randLoc[dire][0]:randLoc[dire][1], minJ+offsetJ:maxJ+offsetJ]
            return cpImage, minI+randLoc[dire][1]-randLoc[dire][0], minJ, maxI, maxJ
        else:
            cpImage[minI:maxI, minJ:maxJ] = image[randLoc[dire][0]:(randLoc[dire][0]+maxI-minI), minJ+offsetJ:maxJ+offsetJ]
            return cpImage, 0, 0, 0, 0
    else:
        offsetI = rd.randint(handPadding[0]-minI, image.shape[0]-handPadding[0]-maxI)
        if maxJ-minJ > randLoc[dire][3]-randLoc[dire][2]: # 区域宽度不够填充
            cpImage[minI:maxI, minJ:(minJ+randLoc[dire][3]-randLoc[dire][2])] = image[minI+offsetI:maxI+offsetI, randLoc[dire][2]:randLoc[dire][3]]
            return cpImage, minI, minJ+randLoc[dire][3]-randLoc[dire][2], maxI, maxJ
        else:
            cpImage[minI:maxI, minJ:maxJ] = image[minI+offsetI:maxI+offsetI, randLoc[dire][2]:(randLoc[dire][2]+maxJ-minJ)]
            return cpImage, 0, 0, 0, 0


# function: 根据选区的背景图像，对其进行翻转生成更多的背景图像
# params: 图像源路径，图像目标路径
# return: null
def rotateImages(srcPath, dstPath):
    i = 0
    imagePaths, _ = imgTool.ReadPath(srcPath)
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        # 水平翻转
        horizonImage = image[:,::-1]
        # 垂直翻转
        verticalImage = image[::-1]
        cv2.imwrite(dstPath + "/" + str(i) + ".png", image)
        i += 1
        cv2.imwrite(dstPath + "/" + str(i) + ".png", horizonImage)
        i += 1
        cv2.imwrite(dstPath + "/" + str(i) + ".png", verticalImage)
        i += 1

if __name__ == '__main__':
    srcPath = "/home/guest01/projects/chromos/utils/chromotest/cla1"
    dstPath = "/home/guest01/projects/chromos/utils/chromotest_result/cla1"
    rotateSrcPath = "/home/guest01/projects/chromos/utils/chromotest/labelme"
    rotateDstPath = "/home/guest01/projects/chromos/utils/chromotest_result/augmentationSample"
    jsonPaths, _ = imgTool.ReadPath(srcPath)
    for jsonPath in jsonPaths:
        data = json.load(open(jsonPath))
        image = img_b64_to_arr(data.get("imageData"))
        if len(image.shape)==3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        instances, minI, minJ, maxI, maxJ = getInstances(image.shape, data["shapes"])
        # 选取染色体区域的上下侧相同宽度填充部分染色体区域
        cpImage = image.copy()
        while minI<maxI and minJ<maxJ:
            cpImage, minI, minJ, maxI, maxJ = fillImageNonChromos(cpImage, minI, minJ, maxI, maxJ)
        filenamePre, _, _, _ = imgTool.parsePath(jsonPath)
        savePath = dstPath + "/" + filenamePre + ".png"
        cv2.imwrite(savePath, cpImage)
    # rotateImages(rotateSrcPath, rotateDstPath)
    
    # lbl, ins = shapes_to_label(image.shape, data["shapes"],label_name_to_value)
    # randRgbs = generateRandRgbs(image, image[mask].shape[0])
    # lblsave("/home/guest01/projects/chromos/utils/lbl.png", lbl)
    # lblsave("/home/guest01/projects/chromos/utils/ins.png", ins)
