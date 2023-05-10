'''Multiple process for inference'''
import random

import torch
import time
from PIL import Image
from torchvision import transforms, models
import numpy as np
import os
from random import choice
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

cnn_model_list = ['alexnet',  'densenet121',  'efficientnet_v2_l', \
                  'googlenet', 'inception_v3',  'mobilenet_v3_small',  'resnet50','vgg16']
yolo_model_list = [ 'yolov5s']
deeplab_model_list = ['deeplabv3_resnet50']
model_list = cnn_model_list + yolo_model_list + deeplab_model_list
model_list = cnn_model_list # for testing with training
infer_image_num = 2000   # infer in one round/episode

def process_image(image_path, image_size, model_name):
    image= Image.open(image_path)
    if len(np.array(image).shape) < 3:  # gray image
        image = image.convert("RGB")  # if gray imange , change to RGB (3 channels)
    if type(image_size) is not tuple: image_size = (image_size, image_size)
    data = image  # input data in type of PIL.image
    if model_name in cnn_model_list or model_name in deeplab_model_list:
        data = transform(image, image_size)
        data = torch.unsqueeze(data, dim=0)  # add a batch_size dimension
    else:  # yolo
        data = data.resize(image_size)
    return data

def work(args):
    model_name, device, image_size, image_folder =  args['arch'], args['device'], args['image_size'], args['image_folder']
    print('inferences start to run... ', args)
    ## process device
    assert device in ['cuda', 'cpu'], 'Device shall be "cuda" or "cpu" !'
    assert not ((not torch.cuda.is_available()) and (
            device == 'cuda')), 'set device of cuda, while it is not available'

    ## load model
    previous_model =''
    if model_name != previous_model:  # if the model is the same as previous one , no need load model again
        model = load_model(model_name, device)
        previous_model = model_name
    # model.to(device)
    model.eval()

    latency_list = []

    with torch.no_grad():
        ## process image
        nn = 0
        for i, image_name in enumerate(os.listdir(image_folder)):
            if nn >infer_image_num: break   # around 2000s to end when apply
            nn+=1
            # print("nnnnnnnnn:", nn)
            try:
                for_model = model_name
                image_path= os.path.join(image_folder, image_name)
                data = process_image(image_path, image_size, for_model)
                ## visualize the image transfered
                # cv_image = np.array(data)
                # # cv_image = cv_image[:, :, ::-1].copy()
                # cv2.imshow('image', cv_image)
                # cv2.waitKey(200)

                if device == 'cpu':
                    start = time.time()
                else:  # cuda
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True)
                    starter.record()  # cuda

                if model_name in cnn_model_list:  # cnn models

                    data = data.to(device)
                    prd = model.forward(data)
                    prd = prd.to('cpu').detach()
                    result = np.argmax(prd, axis=1).numpy()[0]  # transfer result from a tensor to a number

                elif model_name in yolo_model_list:  # YOLO models
                    prd = model(data)
                    result = prd.xyxyn
                elif model_name in deeplab_model_list:
                    img = data.to(device)
                    prd = model.forward(img)
                    result = prd['out']  # prd['out'].shape is [1, 19, 224, 224]
                    # result = visualize_seg(img, prd)  # shape is [3, 244,244], but if overlay processed at server, will consume hundress ms @ server

                if device == 'cpu':
                    latency = (time.time() - start) * 1000  # metrics in ms
                else:  # gpu
                    ender.record()
                    torch.cuda.synchronize()  ###
                    latency = starter.elapsed_time(ender)  # metrics in ms

                    ## vidualize the result of yolo
                    # frame = np.squeeze(prd.render())
                    # cv2.imshow('Window_', frame)
                    # cv2.waitKey(200)
                latency_list.append(latency)
                # print(result)
            except:
                break
            if i > infer_image_num:
                break
        return {'model': model_name, 'latency(ms)': np.mean(latency_list[3:])}

    ## log result

def load_model(model_name, device):
    ## process model
    assert model_name in model_list, f'Input model is not supported, shall be one of {model_list}'
    if model_name in cnn_model_list:
        model_func = 'models.' + model_name
        model = eval(model_func)(pretrained=True) # eval(): transform string to variable or function
        model.to(device)
    elif model_name in yolo_model_list:
        device_yolo ='cpu'  if device == 'cpu' else 0
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, device=device_yolo)
    elif model_name in deeplab_model_list:
        model_func = 'models.segmentation.' + model_name
        # print(model_func)
        model = eval(model_func)(num_classes=19)
        model.to(device)

    return model

def transform(image, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    return transform(image)
def image_folder(root, model):
    if model in cnn_model_list:
        folder = './mini_imagenet/images'  ## mini_imagenet
        folder = os.path.join(root, folder)
        # folder = '/data/datasets/imagenet/ILSVRC2012_img_test_v10102019'  ## imagenet
    elif model in deeplab_model_list:
        folder = './google_street/part1'
        folder = os.path.join(root,folder)
    else:
        folder = './coco/images/test2017'
        folder = os.path.join(root, folder)
    return folder

if __name__ =='__main__':

    # random.seed(3)
    task_num =  4
    latency_list , args_list, res_list = [], [], []
    data_dir = os.path.join(os.environ['HOME'],'./Documents/datasets')

    cnt = 0

    while True:  ## just repeatedly to run untill manually stop
        args_list = []
        pool = mp.Pool(task_num)
        for i in range(task_num):
            args = dict(image_size=224, device='cuda')
            model= choice(model_list)
            args['arch']= model
            args['image_folder'] = image_folder(data_dir, args['arch'])
            args_list.append(args)
        print(args_list)
            # image_folder = os.path.join(data_dir,'Documents/datasets/mini_imagenet/images')
            # args = {'image_size': 224, 'device': 'cuda', 'arch': 'alexnet', 'image_folder': image_folder}


        print('=' * 80)
        print(f'Episode {cnt}.')

        p_list = []
        for i in range(task_num):
            try:
                res_list.append(pool.apply_async(work, args= (args_list[i],)))
                # latency_list.append(result.get())
            except:
                break

        pool.close()
        pool.join()
        #

        for res in res_list:
            # print(res.get())
            latency_list.append(res.get())
        # print(latency_list)

        cnt +=1
