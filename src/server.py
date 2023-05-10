# backup for server 20230306

from PIL import Image
import io
import socket
from util import str2dict, date_time
import time
import numpy as np
import re
import cv2
import pickle
import torch
from torchvision import transforms, models
from util import logger_by_date, visualize_seg
import os
import warnings
warnings.filterwarnings("ignore")


print_interval =  1000 ## to change this value to change the result displaying frequency on the screen
cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large']
model_list = cnn_model_list + yolo_model_list + deeplab_model_list
root = os.environ['HOME']

def load_model(model_name, device):
    ## process model
    assert model_name in model_list, f'Input model is not supported, shall be one of {model_list}'
    if model_name in cnn_model_list:
        model_func = 'models.' + model_name
        model = eval(model_func)(pretrained=True) # eval(): transform string to variable or function
    elif model_name in yolo_model_list:
        device ='cpu'  if device == 'cpu' else 0
        # model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, device=device_yolo)
        local_source = os.path.join(root, './.cache/torch/hub/ultralytics_yolov5_master')
        assert local_source, 'Model does not exist please run download_models.py to download yolov5 models first.'
        model = torch.hub.load(local_source, model_name, source='local', pretrained=True,
                               device=device)  ## model is loaded locally
    elif model_name in deeplab_model_list:
        model_func = 'models.segmentation.' + model_name
        model = eval(model_func)(num_classes=19)
    else: return -1

    return model

def transform(image, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    return transform(image)

def infer(model, model_name, data, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
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
            result = prd['out'] # prd['out'].shape is [1, 19, 224, 224]
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

    return result, latency

global cnt # to count the combination # conducted
cnt = 0
def work(ip,port):
    global cnt
    print(ip,port)
    s = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip, port))
    s.listen(5)
    reply =''
    previous_model = ''
    # print('======== episode ', cnt, ': ==========')
    cnt +=1
    # print('Server is listening...')
    # print(f'Print result every {print_interval} records.','\n')
    while True:
        conn, addr = s.accept()
        # print(f'Received request from {addr} !')
        header = conn.recv(1024)
        try:
            format, args_str = header.decode().split('|')
            args = str2dict(args_str)
            conn.send(b'ok')
        except:
            print('Fail to connect !')
            conn.close()
            continue
        # msg = ('Start...').encode()
        print(args)
        cnt = 0
        while True:
            # print('checking checking msg from client...', reply)
            if reply == 'continuedone'  or reply =='done':  # msg indicates task ends from client
                print(' Job from client is done!\n','='*70,'\n'*3,'Server is listening.....')
                break
            elif reply == 'continue' or reply =='':   # normal condition
                file_info = conn.recv(1024).decode()
                # print('aaa:', file_info)
            else:  ## i.e. file_len|file_name or continuefile_len|file_name
                try: ## solve that can't recognize "done" msg due to packets sticking issue
                    file_info = re.findall(r'([0-9].+)', reply)[0]  ## parse the "continue" msg, which for avoiding blocking (e.g. file_info is 284532|000000000019.jpg)
                    # print('fffffff', file_info)
                except: break

            try:  ## solve can't recognize "done" msg due to packets sticking issue, the last packets of file_info goes ahead of "continue"
                data_len, file_name = file_info.split('|')  ## parse header with file length
            except:
                print(' Job from client is done.\n','=' * 70, '\n\n\nServer is listening.....')
                break

            args['file_name'] = file_name
            model_name, device, image_size = args['arch'], args['device'], args['image_size']
            sub_folder= args['comb_id'] if 'comb_id' in args.keys() else ''

            if model_name !=previous_model :  # if the model is the same as previous one , no need load model again
                model=load_model(model_name, device)
                previous_model = model_name

            ## process device
            assert device in ['cuda', 'cpu'], 'Device shall be "cuda" or "cpu" !'
            assert not ((not torch.cuda.is_available()) and (
                        device == 'cuda')), 'set device of cuda, while it is not available'
            msg = 'start to recieve transfered file from client'.encode() ##
            conn.send(msg)
            if data_len and file_name:
                work_start = time.time()
                # newfile = open(os.path.join(dir, file_name), 'wb')  # save file transfered from server
                file = b''
                data_len = int(data_len)
                get = 0
                while get < data_len: # recieve data
                    data = conn.recv(data_len//2)
                    file += data  ## binary code
                    get += len(data)

                if cnt % print_interval == 0:
                    print(f' {cnt}: File name :{file_name}, {data_len} bytes to transfer, recieved {len(file)} bytes.')
                if file:  # file is in format of binary byte
                    # newfile.write(file[:])  # save file transfered from server
                    # newfile.close()  # save file transfered from server

                    ## process image
                    image = Image.open(io.BytesIO(file))  # convert binary bytes to PIL image in RAM, i.e. 'PIL.JpegImagePlugin.JpegImageFile'
                    if len(np.array(image).shape) < 3:  # gray image
                        image = image.convert("RGB")  # if gray imange , change to RGB (3 channels)
                    if type(image_size) is not tuple: image_size =(image_size,image_size)
                    data = image  # input data in type of PIL.image
                    if model_name in cnn_model_list or model_name in deeplab_model_list:
                        data = transform(image, image_size)
                        data = torch.unsqueeze(data, dim=0)  # add a batch_size dimension
                    else: # yolo
                        data = data.resize(image_size)
                    ## visualize the image transfered
                    # cv_image = np.array(data)
                    # # cv_image = cv_image[:, :, ::-1].copy()
                    # cv2.imshow('image', cv_image)
                    # cv2.waitKey(200)

                    # ## process inference
                    result, latency = infer(model, model_name, data, device)

                    ### send back the result to client

                    # ## option 1:send back the result to client, old version of sending small # of result value
                    # result_cont = pickle.dumps({'file_name': file_name, 'latency_server(ms)': latency, 'result': result})
                    # conn.send(result_cont)

                    ## option 2: send result to client
                    result_cont = pickle.dumps({'file_name': file_name, 'latency_server(ms)': latency, 'result': result})  # serialize the result for sending back to client
                    result_cont_size = len(result_cont)

                    conn.send(str(result_cont_size).encode())
                    msg = conn.recv(1024).decode()
                    # print(result_cont_size)
                    # print("shall get msg of result size recieved., actual: ", msg)  # result size recieved

                    p = 0
                    while p < result_cont_size:
                        to_send = result_cont[p:p + result_cont_size // 2]
                        conn.send(to_send)
                        p += len(to_send)

                    if cnt % print_interval ==0:
                        print(f' {cnt}: File name: {file_name}, Result: {result}, Latency: {latency} ms.\n')

                # save log
                dt, tm = date_time()  # catch current datatime
                col = ['work_start', 'infer_model_name', 'train_model_name', 'image_size', 'device', 'file_name', 'latency']
                data_in_row = [work_start, model_name, args['train_model_name'], args['image_size'], device, args['file_name'], latency]
                logger_prefix = 'azure-multi-infer_log_server_' + str(args['request_rate']) + 'rps_' +' train_'+args['train_model_name']+'+infer_'+model_name+'_'
                log_dir = os.path.join(os.environ['HOME'], r'./Documents/profile_train_infer/result/log/infer_server', dt, sub_folder)
                if not os.path.exists(log_dir): os.makedirs(log_dir)
                logger_by_date(col, data_in_row, log_dir, logger_prefix)

            reply = conn.recv(1024).decode()
            cnt += 1
        reply = conn.recv(1024).decode()  # to recieve notice when client starts a new task

if __name__ =='__main__':
    ip, port = '192.168.85.71', 54100  ## high speed switch
    # ip , port = '128.226.119.71', 51400
    ip, port = '0.0.0.0', 5400
    work(ip, port)
