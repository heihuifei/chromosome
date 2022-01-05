'''
Author: hxp
Introduction: 对染色体原图进行规则切割，并采用传统边缘检测方法提取不同类别的单条染色体生成数据集
Time: 2021.11.13
'''
import queue
import os
import math
import uuid
import json
import os.path as osp

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpy.random.mtrand import choice
import pandas as pd

import image_tool as imgTool
import generate_segmentation_dataset as gsd

label_name_to_value = {"_background_": 0, "chromos": 1}


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


# function: 根据图像像素值分布，按概率分布随机生成cnt个(r,g,b)值
# params: 图像，生成(r,g,b)个数
# return: [[r1,g1,b1] [r2,g2,b2]...] (type=array)
def generateRandRgbs(image, cnt):
    # 将image数组变型为的rgb像素值数组
    imageRgb = image.reshape(image.shape[0] * image.shape[1], 3)
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


# function: 根据shapes字段内容获取图像所有实例的mask的bool数组
# params: 图像尺寸，shapes字段内容
# return: [mask0, mask1...] (type=array)
def getInstances(img_shape, shapes):
    instances = []
    for shape in shapes:
        points = shape["points"]
        shape_type = shape.get("shape_type", None)
        mask = shape_to_mask(img_shape[:2], points, shape_type)
        # print(mask, mask.shape, type(mask))
        instances.append(mask)
    return instances


if __name__ == '__main__':
    image = cv2.imread(
        "/home/guest01/projects/chromos/utils/18-Y2087.145.O.JPG")
    data = json.load(
        open("/home/guest01/projects/chromos/utils/18-Y2087.145.O.json"))
    lbl, ins = shapes_to_label(image.shape, data["shapes"],
                               label_name_to_value)
    # 获得的是一个二维bool数组，label=1的像素点值为true，1代表chromos标签
    mask = lbl == 1

    # instances = getInstances(image.shape, data["shapes"])
    # for v in instances:
    #     print(v, v.shape, image[v].shape[0])

    # randRgbs = generateRandRgbs(image, image[mask].shape[0])
    # print(randRgbs, randRgbs.shape, type(randRgbs))

    lblsave("/home/guest01/projects/chromos/utils/lbl.png", lbl)
    lblsave("/home/guest01/projects/chromos/utils/ins.png", ins)
    cv2.imwrite("/home/guest01/projects/chromos/utils/mask.png", image)
