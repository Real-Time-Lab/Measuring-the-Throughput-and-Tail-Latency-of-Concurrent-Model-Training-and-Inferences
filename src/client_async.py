import pickle
import socket
import os
import time
import datetime
import numpy as np
from util import dict2str, logger_by_date, date_time
import threading
import mp_exception as mp_new
import asyncio
import re

cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large']


def open_file(file_path):
    data = b''
    try:
        file = open(file_path, 'rb')
        data = file.read()
    except:
        print(f' Error ! Get file path {file_path} ')
    else:
        file.close()
        return data

def image_folder(data_dir, model):
    if model in cnn_model_list:
        # folder = './mini_imagenet/images'
        folder = './coco/images/test2017'
    elif model in deeplab_model_list:
        # folder = './google_street/part1'
        folder = './coco/images/test2017'
    else:
        folder = './coco/images/test2017'
    return os.path.join(data_dir,folder)

def gen_poission_interval(request_rate):
    poisson = np.random.poisson(request_rate, 60)
    # print(poisson)
    interval_list = []
    for p in poisson:
        for i in range(p):
            interval_list.append(1 / p)  # in second metric. e.g. 0.16667
    print('Instance amount :', len(interval_list))
    return interval_list

async def handle(ip,port, comb_id, dir, file_name, args, accum_interval):
    await asyncio.sleep(accum_interval)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip , port))
    args['comb_id']=comb_id # this is extra argument for compatible of multiple inference running
    args_str = dict2str(args)

    ## s.send(b'start')
    data_format = 'jpg'
    header = data_format + '|' + args_str
    s.send(header.encode())
    reply = s.recv(1024)

    latency_list =[]
    start = time.time()
    if reply.decode() == 'ok':
        work_start = time.time()
        data = open_file(os.path.join(dir, file_name))
        data_len = len(data)
        file_info = str(data_len) + '|' + file_name
        s.send(file_info.encode())
        msg = s.recv(1024) # for unblocking, r"start receiving ..."
        # print('recived file info:', msg.decode())

        now = time.time()
        # print('Latency:', (now-start)*1000 , 'ms.')
        latency = (now-start)*1000
        latency_list.append(latency)
        start = now
        p = 0
        while p < data_len:
            to_send = data[p:p + data_len // 2]
            s.send(to_send)
            p += len(to_send)

        # ## option 1: receive result from server,  old version of sending small # of result value
        # result_from_server = pickle.loads(s.recv(1024))  # old: deserialize the result from server

        # ## option 2: receive result from server
        # s.send('Ready to recieve results size from server.'.encode())
        result_cont_size = s.recv(1024).decode()  # result file info from server
        result_cont_size = int(result_cont_size)
        # print(result_cont_size)
        s.send('Result size recieved'.encode())
        if result_cont_size > 0:
            result_cont = b''
            get = 0
            while get < result_cont_size:  # recieve data
                data = s.recv(result_cont_size // 2)
                result_cont += data  ## binary code
                get += len(data)
        result_from_server =  pickle.loads(result_cont)

        s.send(b'continue')

        # log the instance
        dt, tm = date_time() # catch current datatime
        col = ['work_start', 'infer_model_name', 'train_model_name','image_size', 'device', 'file_name', 'latency', 'request_rate']
        data_in_row = [work_start, args['arch'], args['train_model_name'], args['image_size'], args['device'], file_name, latency, args['request_rate']]
        logger_prefix = 'infer_log_client_' + str(args['request_rate']) +'rps_' +  ' train_' + args[
            'train_model_name'] + '+infer_' + args['arch'] + '_'
        log_dir = os.path.join(os.environ['HOME'], r'./Documents/profile_train_infer/result/log/infer_client', dt, comb_id)
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        logger_by_date(col, data_in_row, log_dir, logger_prefix)

        s.send(b'done')
        s.close()
        # print('Latency(ms): ', latency_list[:])

async def work(ip, port, comb_id, img_folder, file_list, args, request_interval_list, print_interval):
    tasks = []
    accum_interval = 0
    for i in range(len(request_interval_list)):
        if i >= len(file_list) - 1: break  # if out of file_list, quit
        file_name = file_list[i]
        accum_interval += request_interval_list[i]
        tasks.append(asyncio.ensure_future(
            handle(ip, port, comb_id, img_folder, file_name, args, accum_interval)))
    await asyncio.wait(tasks)

def main(ip, port, comb_id, request_rate_list, arch_list, train_model_name, print_interval):  # if single inference, comb_id =NULL
    np.random.seed(6)
    # print(f'Print status every {print_interval} records.')
    print('Start time: ', datetime.datetime.now())
    root = os.environ['HOME']
    data_dir = os.path.join(root, r'./Documents/datasets/')

    for request_rate in request_rate_list: ## e.g. [10,20] or [10]
        if request_rate == 0:
            with open('/home/anurag/Documents/exps/profile-training-inference-cowork-71/traces/delays_to_be_used_for_AZURE_TRACES_experiments.csv', 'r') as f:
                request_interval_list = f.read().splitlines()
                request_interval_list = [float(x) for x in request_interval_list]
        else:
            request_interval_list = gen_poission_interval(request_rate)
        for arch in arch_list: ## e.g. ['yolov5s', 'resnet50']
            args = dict(request_rate = request_rate, arch=arch, train_model_name = train_model_name, device='cuda', image_size=224)  # deeplabv3_resnet50
            # img_folder = os.path.join(root, r'./Documents/exps/profile-training-inference-cowork-71/datasets/coco/images/test2017') ## Anurugs
            img_folder = image_folder(data_dir, args['arch'])  # select dataset to fit models
            file_list = []
            for file in os.listdir(img_folder):
                if re.findall(r'.(jpg)', file)[0] in ['jpg', 'png', 'bmp', 'jpeg']: file_list.append(
                    os.path.join(img_folder, file))

            loop = asyncio.get_event_loop()
            loop.run_until_complete(work(ip, port, comb_id, img_folder, file_list,\
                                         args, request_interval_list, print_interval))
            loop.close()

            # time.sleep(1)

if __name__ == '__main__':
    ip, port = '127.0.0.1', 5400  ## change to your real ip address

    print_interval = 1000  # to change this value to change the result displaying frequency on the screen

    arch_list = ['vgg16', 'resnet50', 'alexnet', 'densenet121', 'efficientnet_v2_l', 'googlenet', 'inception_v3+', 'mobilenet_v3_small' ]
    ## train_model_list = ['none', 'resnet152_32', 'vgg16_64']
    train_model_name = 'none'  # manually change the name here , batch size as well!!
    request_rate_list = [40]
    comb_id=''  # to be compatable with multiple model infering
    ######################## run the inference combinations ########################
    main(ip, port, comb_id, request_rate_list, arch_list, train_model_name, print_interval )
    ################################################################################
