import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os


device = 'cuda:0'


# function: 模型对图像进行预测并可视化
# params: 图像路径，预测结果存储路径，配置文件，离线模型
# return: null
def showImage(imagesPath, savePath, config, model):
    for filename in os.listdir(imagesPath):
        img = os.path.join(imagesPath, filename)
        result = inference_detector(model, img)
        print("This is result: ", len(result), result)
        out_file = os.path.join(savePath, filename)
        # 可在mmdet/api/inference中查看相关参数
        model.show_result(
            img,
            result,
            score_thr=0.6,
            out_file=out_file,
            text_color = None)
        # show_result_pyplot(model, img, result, score_thr=0.6)


def strongContrat(imagePath, savePath):
    img = cv2.imread(imagePath)
    img_tmp = cv2.normalize(img, dst=None, alpha=255, beta=10, norm_type=cv2.NORM_MINMAX)
    img_norm=cv2.normalize(img_tmp, dst=None, alpha=300, beta=0, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite(savePath, img_norm)

if __name__ == '__main__':
    # 需要加载的测试图片的文件路径
    imagesPath = "/home/guest01/projects/chromos/dataset/segmentation_dataset/val_origin23and37images_annotated_coco/JPEGImages"
    # 保存测试图片的路径
    savePath = "/home/guest01/projects/chromos/utils/imageTest"
    # 模型配置路径
    config_file = "/home/guest01/projects/chromos/mmdetection/configs/chromos/my_mask_rcnn_convnext_t_p4_w7_mstrain_3x.py"
    # 训练好的模型参数
    checkpoint_file = "/home/guest01/projects/chromos/outputModels/segmentationModels/msrcnn/msrcnn_convnext_t_p4_w7_3x_dist_origin_fake1500/epoch_19.pth"
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    showImage(imagesPath, savePath, config_file, model)
    
    # imagePath = "/home/guest01/projects/chromos/dataset/segmentation_dataset/chromosome_coco_format/chromos/train_origin_77and187/JPEGImages"
    # save = "/home/guest01/projects/chromos/dataset/segmentation_dataset/chromosome_coco_format/chromos/train_origin_77and187clear/JPEGImages"
    # for filename in os.listdir(imagePath):
    #     Path = os.path.join(imagePath, filename)
    #     Save = os.path.join(save, filename)
    #     strongContrat(Path, Save)
    #     # break
