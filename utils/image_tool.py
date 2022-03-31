'''
Author: hxp
Introduction: 基础的图像工具，如获取文件夹下图像文件，对图像进行放缩
Time: 2021-11-13
'''
import os

import cv2


# function：将输入图片保持长宽比放缩图片
# params: 输入图片, 输出图片大小，如(500,500)
# return: 目标图片
def ResizeKeepAspectRatio(imageSrc, dstSize):
    srcH, srcW = imageSrc.shape[:2]
    dstH, dstW = dstSize

    #判断应该按哪个边做等比缩放
    h = dstW * (float(srcH) / srcW)  #按照ｗ做等比缩放
    w = dstH * (float(srcW) / srcH)  #按照h做等比缩放
    h = int(h)
    w = int(w)
    if h <= dstH:
        imageDst = cv2.resize(imageSrc, (dstW, int(h)))
    else:
        imageDst = cv2.resize(imageSrc, (int(w), dstH))
    h_, w_ = imageDst.shape[:2]
    top = int((dstH - h_) / 2)
    down = int((dstH - h_ + 1) / 2)
    left = int((dstW - w_) / 2)
    right = int((dstW - w_ + 1) / 2)
    borderType = cv2.BORDER_CONSTANT
    imageDst = cv2.copyMakeBorder(imageDst,
                                  top,
                                  down,
                                  left,
                                  right,
                                  borderType,
                                  value=(255, 255, 255))
    return imageDst


# description: 根据路径读取目录下所有文件和文件夹
# params: 所需要读取的路径
# return: 所有文件的绝对路径列表, 所有文件夹的绝对路径列表
def ReadPath(path):
    filePaths = []
    dirPaths = []
    # 路径, 文件夹名列表, 文件名列表
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            dirPaths.append(os.path.join(path, dir))
        for file in files:
            filePaths.append(os.path.join(path, file))
        break
    return filePaths, dirPaths


# function: 根据路径在指定路径中创建chromos1, chromos2...目录
# params: 所需要创建目录的路径, 所要创建的文件夹名称后缀
# return: null
def Makedir(path, postfix):
    for i in range(22):
        os.mkdir(path + "./chromo" + str(i) + postfix)
    os.mkdir(path + "./chromoX" + postfix)
    os.mkdir(path + "./chromoY" + postfix)


# function: 根据文件的绝对路径获取文件名
# params: 文件绝对路径
# return: 文件名前缀，文件名后缀，文件名，文件父路径
def parsePath(path):
    (fatherPath, filename) = os.path.split(path)
    (filenamePrefix, filenamePostfix) = os.path.splitext(filename)
    return filenamePrefix, filenamePostfix, filename, fatherPath

# function: 根据目录路径批量修改文件名
# params: 目录路径
# return: null
def Rename(path):
    filePaths, _ = ReadPath(path)
    for filePath in filePaths:
        pre, post, filename, _ = parsePath(filePath)
        newName = path + os.sep + pre + ".small" + post
        os.rename(filePath, newName)

def Delete(path):
    filePaths, _ = ReadPath(path)
    for i in range(len(filePaths)):
        pre, post, filename, _ = parsePath(filePaths[i])
        if pre[0:4]=='fake':
            if int(pre[4:])%2==0 and int(pre[4:])<1000:
                print(pre[4:], " ", filePaths[i])
                os.remove(filePaths[i])

if __name__ == '__main__':
    # 用于测试各个工具函数的正确性
    print(
        parsePath(
            "/home/guest01/projects/chromos/utils/chromotest/classification/161923.054.K.JPG"
        ))
    # Delete("/home/guest01/projects/chromos/dataset/segmentation_dataset/train_origin77and187images_fake1000_withClear_annotated")
    # print(ReadPath("/home/guest01/projects/chromos/utils"))
    Rename("/home/guest01/projects/chromos/dataset/segmentation_dataset/ttt")
