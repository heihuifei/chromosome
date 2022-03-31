'''
Author: hxp
Introduction: 对染色体原图进行规则切割，并采用传统边缘检测方法提取不同类别的单条染色体生成数据集
Time: 2021.11.13
'''
import queue

import cv2
from pylab import *
from json import dumps
import json
from base64 import b64encode

import image_tool as imgTool
import matplotlib.pyplot as plt
import generate_classification_dataset as gcd


class Pixel(object):
    '用于表示像素值及其在图像中的个数'

    def __init__(self, pixelCnt, pixelVal):
        self.pixelCnt = pixelCnt
        self.pixelVal = pixelVal

    def __lt__(self, other):
        if self.pixelCnt != other.pixelCnt:
            return self.pixelCnt > other.pixelCnt
        else:
            return self.pixelVal > other.pixelVal


# function: 生成labelme实例标注格式的json数据
# params: 图像路径, 存储输出路径
# return: json（{"key": "val"}格式数据）
def generateJson(imagePath, savePath):
    image = cv2.imread(imagePath)
    version = "4.5.7"
    flags = {}
    shapes = generateJsonShapes(image)
    imageData = transImage2RawData(imagePath)
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
    filenamePre, _, _, _ = imgTool.parsePath(imagePath)
    jsonSavePath = savePath + "/" + filenamePre + ".json"
    json.dump(annotateJson, open(jsonSavePath, 'w'))


# function: 生成json中shapes字段内容
# params: 图像
# return: json（{"key": "val"}格式数据）
def generateJsonShapes(image):
    # shapes字段包括label, points, group_id, shape_type, flag五个字段
    shapes = []
    label = "chromos"
    shape_type = "polygon"
    group_id = None
    flags = {}
    pointsList = getPointsByConnect(image, 190)
    for points in pointsList:
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
def getPointsByContour(img, thre):
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
    contourPoints = []
    ret, thresh = cv2.threshold(gray, best_thre, 255,
                                cv2.THRESH_BINARY)  #循环动态获取最好的二值图像
    canny = cv2.Canny(thresh, 0, 255)  # 通过canny算子获取该效果较好的二值图像的轮廓
    # 绘制轮廓,第三个参数是轮廓的索引（在绘制单个轮廓时有用。要绘制所有轮廓，请传递-1）
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    point0 = []
    for x in range(0, contours[0].shape[0]):
        point0.append(
            [float(contours[0][x][0][0]),
             float(contours[0][x][0][1])])
    contourPoints.append(point0)
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
                contourPoints.append(point)
    return contourPoints


# function: 根据图像采用连通域获取所有的标注坐标点列表
# params: 图像路径
# return: [[[x, y],[x, y],[x, y]], [[x, y],[x, y]]] (轮廓坐标list)
def getPointsByConnect(image, thre):
    contoursPoints = []
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 源图片, 阈值, 填充色, 阈值类型，根据将小于thre的像素点置0(黑色)或置填充色
    # type=0(小于thre置0, 大于阈值置填充色)，type=1与0相反，type=3(小于thre置0, 大于阈值保持原色)type=4与3相反
    _, threshImage = cv2.threshold(grayImage, thre, 255, 0)
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(threshImage), 8, ltype=cv2.CV_32S)
    for i in range(1, len(stats)):
        # TODO: 判断该连通域是否有效进而完成mask的生成
        if validChromosomeConnect(stats[i]) == 0:
            # 返回为三维坐标点数组，需要对齐进行reshape降维
            chromoContour = dilateConnectMaskContour(image, i, labels)
            contourPoints = chromoContour.reshape((chromoContour.shape[0], 2))
            # contourPoints为array数组格式数据，需要转为list后才能用于构建labelme的json文件
            contoursPoints.append(contourPoints.tolist())
    return contoursPoints


# function: 判断连通域是否为有效的染色体连通域
# params: 连通域stat参数（包括中心点，像素个数等参数）
# return: 连通域有效标记，0表示正常有效，1表示连通域过大,-1表示连通域无效
def validChromosomeConnect(stat):
    if stat[4] >= 50 and stat[4] <= 1500:
        return 0
    elif stat[4] >= 1500 and stat[4] < 5000:
        return 1
    else:
        return -1


# function: 根据图像获取其背景像素值及个数
# params: 图像路径
# return: 图像背景像素值及个数（优先队列返回）
def getImageBackgroudPixel(image):
    # 优先队列默认是小值优先
    Pixels = {}
    topQueue = queue.PriorityQueue()
    imageH, imageW = image.shape
    for h in range(imageH):
        for w in range(imageW):
            cnt = 0
            if image[h][w] in Pixels:
                cnt = Pixels[image[h][w]]
            Pixels[image[h][w]] = cnt + 1
    for pixelVal, pixelCnt in Pixels.items():
        pixel = Pixel(pixelCnt, pixelVal)
        topQueue.put(pixel)
    return topQueue


# function: 根据连通域，对其进行膨胀操作后获取其轮廓
# params: 原图像，单个轮廓
# return: 轮廓坐标
def dilateConnectMaskContour(image, label, labels):
    cpImage = np.zeros(image.shape[:2], 'uint8')
    # 获得的是一个二维bool数组，label=i的像素点值为true，后续用作为坐标索引
    mask = labels == label
    # cpImage[:, :, 0]为二维数组，用[mask]来索引
    cpImage[:, :][mask] = 255
    # 定义3x3的核函数，用于膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 操作图像，核函数，操作次数
    dilateImage = cv2.dilate(cpImage, kernel, 1)
    contours, hierarchy = cv2.findContours(dilateImage, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return np.array(contours[0], dtype=float)


def Plt():
    np.random.seed(0)
    x=np.random.rand(20)
    y=np.random.rand(20)
    colors=np.random.rand(20)
    print(x)
    print(y)
    print(colors)
    area=(50*np.random.rand(20))**2
    plt.scatter([0.1, 0.2, 0.5], [0.3, 0.4, 0.6], c=[0.3595079, 0.43703195, 0.6976312], alpha=0.5, cmap = "nipy_spectral", marker = 'x')
    plt.savefig("/home/guest01/projects/chromos/utils/chromotest/testMat.png")

if __name__ == '__main__':
    srcPath = "/home/guest01/projects/chromos/utils/chromotest/segmentation"
    dstPath = "/home/guest01/projects/chromos/utils/chromotest_result/segmentation"
    imagePaths, _ = imgTool.ReadPath(srcPath)
    for imagePath in imagePaths:
        generateJson(imagePath, dstPath)
        # getPointsByConnect(cv2.imread(imagePath), 200)
        # getImageBackgroudPixel(imagePath)
        pass
    # Plt()
