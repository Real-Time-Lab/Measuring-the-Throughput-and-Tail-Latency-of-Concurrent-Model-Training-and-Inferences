'''
model will be downlaod autometically, at:
cnn and deeplab models:
    ~/.cache/torch/hub/checkpoints

yolo:
    ~/.cache/torch/hub

'''

from torchstat import stat
from torchsummary import summary
from torchvision import models
import torch

cnn_model_list = ['inception_v3']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101','deeplabv3_mobilenet_v3_large']
model_list = cnn_model_list + deeplab_model_list + yolo_model_list

cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet50', 'deeplabv3_mobilenet_v3_large']
model_list = cnn_model_list + yolo_model_list + deeplab_model_list

for model_name in model_list:

    # if model_name in deeplab_model_list:
    #     print(model_name)
    #     model_func = 'models.segmentation.'+model_name
    #     model = eval(model_func)(pretrained=True)
    #     # stat(model,(3,224,448))
    #     print('\n' * 5)
    #
    if model_name in cnn_model_list:
        print(model_name)
        model_func = 'models.' + model_name
        model = eval(model_func)(pretrained=True)
        stat(model,(3,224,224))
        print('\n'*5)

    ## manually: wget -c -t 0 https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
    # if model_name in yolo_model_list:
    #     model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, device =0)
    #     # summary(model,(3,224,224))
    #     print('\n'*5)