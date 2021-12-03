'''
Author: hxp
Introduction: 对染色体原图进行规则切割，并采用传统边缘检测方法提取不同类别的单条染色体生成数据集
Time: 2021.11.13
'''
import queue
import os

import cv2
import numpy as np

import image_tool as imgTool

# 用于表示从1~22, X, Y类别染色体的面积特征
classAreaCharacter = [
    24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
    4, 3, 2, 1
]
# 用于表示从1~22, X, Y染色体在分类图中的位置, 产前检查染色体分类图片
classLocation1 = [
    # [x1, x2, y1, y2]，其中x为左上角向左下角方向的x轴
    [0, 0, 0, 0],
    [50, 260, 0, 200],  # 1号染色体位置
    [50, 260, 190, 390],
    [50, 260, 360, 560],
    [50, 260, 650, 850],
    [50, 260, 830, 1030],  # 5号染色体位置
    [290, 450, 0, 180],  # 6号染色体
    [290, 450, 160, 320],
    [290, 450, 280, 440],
    [290, 450, 410, 570],
    [290, 450, 580, 730],
    [290, 450, 710, 870],
    [290, 450, 840, 1010],  # 12号染色体
    [490, 615, 5, 160],  # 13号染色体
    [490, 615, 140, 300],
    [490, 615, 280, 440],
    [490, 615, 580, 730],
    [490, 615, 710, 870],
    [490, 615, 840, 1010],  # 18号染色体
    [680, 808, 5, 160],  # 19号染色体
    [680, 808, 140, 300],
    [680, 808, 350, 540],
    [680, 808, 530, 680],  # 22号染色体
    [680, 808, 710, 870],  # X号染色体
    [680, 808, 840, 1010]  # Y号染色体
]

# 用于表示从1~22, X, Y染色体在分类图中的位置, 血液病检查染色体分类图片
classLocation = [
    # [x1, x2, y1, y2]，其中x为左上角向左下角方向的x轴
    [0, 0, 0, 0],
    [50, 260, 0, 200],  # 1号染色体位置
    [50, 260, 190, 390],  # 2号染色体
    [50, 260, 360, 560],  # 3号染色体
    [50, 260, 650, 850],  # 4号染色体
    [50, 260, 830, 1030],  # 5号染色体位置，第一行
    [290, 450, 0, 180],  # 6号染色体
    [290, 450, 160, 320],  # 7号染色体
    [290, 450, 300, 460],  # 8号染色体
    [290, 450, 440, 600],  # 9号染色体
    [290, 450, 570, 730],  # 10号染色体
    [290, 450, 710, 880],  # 11号染色体
    [290, 450, 860, 1030],  # 12号染色体，第二行
    [490, 620, 0, 180],  # 13号染色体
    [490, 620, 150, 320],  # 14号染色体
    [490, 620, 290, 460],  # 15号染色体
    [490, 620, 570, 730],  # 16号染色体
    [490, 620, 710, 880],  # 17号染色体
    [490, 620, 850, 1030],  # 18号染色体，第三行
    [670, 815, 0, 160],  # 19号染色体
    [670, 815, 140, 300],  # 20号染色体
    [670, 815, 340, 510],  # 21号染色体
    [670, 815, 480, 660],  # 22号染色体
    [670, 815, 680, 890],  # X号染色体
    [670, 815, 860, 1030]  # Y号染色体，第四行
]


# 定义可疑染色体类继承object基类并重写__lt__方法实现优先队列自定义排序
class TChromosome(object):
    '用于表示可疑染色体及其特征'

    def __init__(self, area, seq):
        self.area = area
        self.seq = seq

    def __lt__(self, other):
        return self.area > other.area


# function: 根据位置将原始图片的每对染色体切割为一张图片
# params: 输入图片, 处理染色体序号, 输出图片大小, 输出保存路径
# return: null
def segmentDoubleChromosome(imagePath, seq, dstSize, savePath):
    imageSrc = cv2.imread(imagePath)
    # ! 用于查看各个类别染色体的切割位置是否正确
    # [x1:x2, y1,y2]，其中x为左上角向左下角方向的x轴
    for i in range(1, 25):
        chromoiDouble = imageSrc[classLocation[i][0]:classLocation[i][1],
                                 classLocation[i][2]:classLocation[i][3]]
        chromoiDoubleResize = imgTool.ResizeKeepAspectRatio(
            chromoiDouble, dstSize)
        chromoiSingle1, chromoiSingle2 = segmentSingleChromosomeByConnect(
            chromoiDoubleResize, 230, i, savePath)
        chromoiDir = savePath + "/chromo" + str(i) + "/"
        if not os.path.exists(chromoiDir):
            os.makedirs(chromoiDir)
        if i == 23:
            cv2.imwrite(chromoiDir + "chromoX_" + str(seq) + ".jpg",
                        chromoiSingle1)
            cv2.imwrite(chromoiDir + "chromoX_" + str(seq + 1) + ".jpg",
                        chromoiSingle2)
        elif i == 24:
            cv2.imwrite(chromoiDir + "chromoY_" + str(seq) + ".jpg",
                        chromoiSingle1)
            cv2.imwrite(chromoiDir + "chromoY_" + str(seq + 1) + ".jpg",
                        chromoiSingle2)
        else:
            cv2.imwrite(
                chromoiDir + "chromo" + str(i) + "_" + str(seq) + ".jpg",
                chromoiSingle1)
            cv2.imwrite(
                chromoiDir + "chromo" + str(i) + "_" + str(seq + 1) + ".jpg",
                chromoiSingle2)


# function: 通过边缘检测填充掩膜对成对染色体图片提取出单条染色体并保存
# params：输入图片, 二值化阈值, 染色体类别, 输出保存路径
# return: image1, image2
def segmentSingleChromosomeByContour(image, thre, chromoClass, savePath):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 源图片, 阈值, 填充色, 阈值类型，根据将小于thre的像素点置0(黑色)或置填充色
    # type=0(小于thre置0, 大于阈值置填充色)，type=1与0相反，type=3(小于thre置0, 大于阈值保持原色)type=4与3相反
    _, threshImage = cv2.threshold(grayImage, thre, 255, 0)
    # 通过canny算子获取该效果较好的二值图像的轮廓
    cannyImage = cv2.Canny(threshImage, 0, 255)
    # 绘制轮廓,第三个参数是轮廓的索引（在绘制单个轮廓时有用。要绘制所有轮廓，请传递-1）
    contours, hierarchy = cv2.findContours(cannyImage, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # 获取到前两大面积的外侧轮廓编号，即单条染色体的外侧轮廓
    contourIndex1, contourIndex2 = top2Contours(chromoClass, contours)
    contourImage1 = fillContourMask(image, contours[contourIndex1])
    contourImage2 = fillContourMask(image, contours[contourIndex2])
    # TODO: 需要确定将图片输出到什么位置，如何命名
    return contourImage1, contourImage2


# function: 通过对二值图像连通域检测提取出单条染色体并保存
# params：输入图片, 二值化阈值, 染色体类别, 输出保存路径
# return: image1, image2
def segmentSingleChromosomeByConnect(image, thre, chromoClass, savePath):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 源图片, 阈值, 填充色, 阈值类型，根据将小于thre的像素点置0(黑色)或置填充色
    # type=0(小于thre置0, 大于阈值置填充色)，type=1与0相反，type=3(小于thre置0, 大于阈值保持原色)type=4与3相反
    _, threshImage = cv2.threshold(grayImage, thre, 255, 0)
    # 使用connectedComponentsWithStats对二值图像检测连通域，检测区域像素值>0
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(threshImage), 8, ltype=cv2.CV_32S)
    connectIndex1, connectIndex2 = top2Connects(chromoClass, stats)
    connectImage1 = connectMask(image, connectIndex1, labels)
    connectImage2 = connectMask(image, connectIndex2, labels)
    return connectImage1, connectImage2


# function: 找出最符合染色体特征的两条外侧轮廓编号
# params: 切割染色体类别, 输入可疑染色体列表
# return: 最大面积/周长轮廓编号，第二大面积/周长轮廓编号
def top2Contours(chromoClass, contours):
    # 使用优先队列存储最符合染色体特征的轮廓选取前两名
    # TODO: 根据不同特征设计轮廓评分模型，修改ChromosomeContour的实例变量排序规则
    top1Index = -1
    top2Index = -1
    topQueue = queue.PriorityQueue()
    for i in range(len(contours)):
        if validContour(chromoClass, contours[i]):
            tChromosome = TChromosome(cv2.contourArea(contours[i]), i)
            topQueue.put(tChromosome)
    if not topQueue.empty():
        top1Index = topQueue.get().seq
    if not topQueue.empty():
        top2Index = topQueue.get().seq
    return top1Index, top2Index


# function: 找出最符合染色体特征的两个连通域编号
# params: 切割染色体类别, 输入可疑染色体列表
# return: 最大像素点个数连通域编号，第二大像素点个数连通域编号
def top2Connects(chromoClass, stats):
    # 使用优先队列存储最符合染色体特征的轮廓选取前两名
    # TODO: 根据不同特征设计轮廓评分模型，修改ChromosomeContour的实例变量排序规则
    top1Index = -1
    top2Index = -1
    topQueue = queue.PriorityQueue()
    for i in range(1, len(stats)):
        if validConnect(chromoClass, stats[i]):
            tChromosome = TChromosome(stats[i][4], i)
            topQueue.put(tChromosome)
    if not topQueue.empty():
        top1Index = topQueue.get().seq
    if not topQueue.empty():
        top2Index = topQueue.get().seq
    return top1Index, top2Index


# function: 根据面积/周长等特征判断轮廓是否符合染色体特征
# params: 轮廓类别, 单个轮廓
# return: 单个可以染色体是否有效
def validContour(chromoClass, chromo):
    # 在当前的图像尺寸下进行切割，使用其面积大小和指定阈值对比判断是否有效
    if cv2.contourArea(chromo) > 500:
        return True
    else:
        return False


# function: 根据总像素点特征判断连通域是否符合染色体特征
# params: 染色体类别, 单个可疑染色体
# return: 单个可以染色体是否有效
def validConnect(chromoClass, chromo):
    # 在当前的图像尺寸下进行切割，使用其面积大小和指定阈值对比判断是否有效
    return True


# function: 根据轮廓边缘，填充其内部并和原图与运算后实现掩膜单条染色体
# params: 原图像，单个轮廓
# return: 根据轮廓获取的原图像掩膜结果
def fillContourMask(image, contour):
    # 创建和imageSrc同尺寸画布, 均为0像素值无颜色，纯黑色
    cpImage = np.zeros(image.shape, 'uint8')
    # 填充轮廓内部区域得到掩膜, pts为一个数组, 数组元素为二维数组，并需要使用[]括起来
    cv2.fillPoly(cpImage, contour, (255, 255, 255))
    #对二值图进行取反操作，便于后续掩膜提取单条染色体且背景为白色
    maskedImage = cv2.add(image, cv2.bitwise_not(cpImage))
    return maskedImage


# function: 根据连通域边缘，填充其内部并和原图与运算后实现掩膜单条染色体
# params: 原图像，当前连通域的标记label，带连通域标记的图像
# return: 根据label标记获取的原图像掩膜结果
def connectMask(image, label, labels):
    cpImage = np.zeros(image.shape, 'uint8')
    # 获得的是一个二维bool数组，label=i的像素点值为true，后续用作为坐标索引
    mask = labels == label
    # cpImage[:, :, 0]为二维数组，用[mask]来索引
    cpImage[:, :, :][mask] = (255, 255, 255)
    maskedImage = cv2.add(image, cv2.bitwise_not(cpImage))
    return maskedImage


# function: 用于查看切割位置是否准确矫正location数组
# params: 原图像，染色体类别，输出图像位置
# return: null
def showSegLocation(image, i, savePath):
    double = image[classLocation[i][0]:classLocation[i][1],
                   classLocation[i][2]:classLocation[i][3]]
    doubleResize = imgTool.ResizeKeepAspectRatio(double, (500, 500))
    cv2.imwrite(savePath + "/show.png", doubleResize)


if __name__ == '__main__':
    srcPath = "/home/guest01/projects/chromos/utils/chromotest/classification"
    dstPath = "/home/guest01/projects/chromos/utils/chromotest_result"
    imagePaths, _ = imgTool.ReadPath(srcPath)
    seq = 1
    for imagePath in imagePaths:
        segmentDoubleChromosome(imagePath, seq, (500, 500), dstPath)
        seq += 2

    # 用于查看切割位置是否准确矫正location数组
    # showSegLocation(cv2.imread(srcPath+"/161923.054.K.JPG"), 1, dstPath)
