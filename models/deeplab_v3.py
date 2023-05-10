import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation
import multiprocessing as mp
import argparse
from tqdm import tqdm
import sys
sys.path.append("../datasets")
sys.path.append("../src")
import cityscapes
import numpy as np
import cv2
from util import visualize_seg

parser =  argparse.ArgumentParser()
# parser.add_argument('--root', metavar = 'root', default= '/home/royliu')
parser.add_argument('--dataset', metavar = 'data_dir', default= './Documents/datasets/cityscapes')
parser.add_argument('--arch', metavar = 'arch', default = 'deeplabv3_resnet50', help ='e.g. segmentation.deeplabv3_resnet101')
args = parser.parse_args()
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101','deeplabv3_mobilenet_v3_large']
root = os.environ['HOME']

def work_train(config, queue): # call train()
    ## initialize the arguments
    # queue.put(dict(process='deeplabv3_train'))  # for queue exception of this process
    args = parser.parse_args()
    start = time.time()
    # update argparse' args with the new_args from parent process (main)

    for key, value in config.items():
        vars(args)[key] = value  # update args
    print('Train starts .........', args)
    data_dir = os.path.join(root, args.dataset)
    epoch = args.epochs
    batch_size = args.batch_size
    num_workers = args.workers
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device
    print('Device:', device)
    assert args.arch in deeplab_model_list, f'Model \"{args.arch}\" is not recognized!'

    # model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=19)
    # model = eval(model_func)(num_classes=19)
    # model_func = 'torchvision.models.' + args.arch  # load parser_args's arch value before overiding by config

    model_func = 'torchvision.models.segmentation.' + args.arch
    model = eval(model_func)(num_classes=19)

    train_loader = torch.utils.data.DataLoader(cityscapes.DataGenerator(data_dir, split='train'), \
                                               batch_size= batch_size, num_workers= num_workers)
    val_loader = torch.utils.data.DataLoader(cityscapes.DataGenerator(data_dir, split='val'), \
                                             batch_size= batch_size, num_workers= num_workers)

    train(model, train_loader, val_loader, epoch, device, batch_size)
    print('Training Done')
    dur = time.time()-start
    print('Traingin duration: ', dur, ' sec.')
    queue.put(dict(duration_sec = dur))

def train(model, train_loader, val_loader, num_epochs, device, batch_size):  # called by work_train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    device = device

    '''
    Roy: to solve exception when batch size =1, since Batch normalization computes: y = (x - mean(x)) / (std(x) + eps) = 0 then.
    call model.eval() to disable further training. This stops BatchNorm layers from updating their mean and variance
    '''
    if batch_size == 1:
        model.eval().to(device)
    else:
        model.train().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        tr_loss = []
        val_loss = []
        print('Epoch {}/{}'.format(epoch, num_epochs))

        for img, masks in tqdm(train_loader):
            # inputs = torch.tensor(img).to(device)
            # masks = torch.tensor(masks).to(device)
            inputs = img.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # y_pred = outputs['out']
            # y_true = masks
            # loss = criterion(y_pred.float(), y_true.float())
            loss = criterion(outputs['out'], masks)
            loss.backward()
            tr_loss.append(loss)
            optimizer.step()
            # break

        print(f'Train loss: {torch.mean(torch.Tensor(tr_loss))}')

        # for sample in tqdm(val_loader):
        #     if sample['image'].shape[0] == 1:
        #         break
        #     inputs = sample['image'].to(device)
        #     masks = sample['mask'].to(device)
        for img, masks in tqdm(val_loader):
            inputs = img.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            y_pred = outputs['out']
            y_true = masks
            # loss = criterion(y_pred.float(), y_true.long())
            loss = criterion(y_pred.float(), y_true.float())
            val_loss.append(loss)
            optimizer.step()
            # break

        print(f'Validation loss: {torch.mean(torch.Tensor(val_loss))}')

    return model

def work_infer(config, pipe, queue):
    for key, value in config.items():
        vars(args)[key] = value  # update args with config passed
    print(vars(args))

    con_a,con_b = pipe
    con_a.close()
    queue.put(dict(process= 'deeplab_infer'))  # for queue exception of this process
    model_name= args.arch
    batch_size = args.batch_size
    img_size = args.image_size
    device = args.device
    dataset = args.dataset
    data_dir = os.path.join(root, args.dataset)
    num_workers = args.workers

    test_loader = torch.utils.data.DataLoader(cityscapes.DataGenerator(data_dir, split='test'), \
                                             batch_size= batch_size, num_workers= num_workers)

    # model_func = 'models.' + model_name
    # model = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=19)
    model_func = 'torchvision.models.segmentation.' + model_name
    model = eval(model_func)(num_classes=19)

    assert not ((not torch.cuda.is_available()) and (device =='cuda')), 'set device of cuda, while it is not available'
    model.to(device)
    model.eval()

    latency_total, latency, count = 0, 0, 0
    latency_list = []
    quit_while = False

    print()
    print('Infer starts......... ', args)
    while True:  # infer repeatedly
        ######### Below is for stoping infer after receiving msg from cnn_train.py ############
        with torch.no_grad():

            # time.sleep(10)  # sleep 50 waiting for trainin
            for image, label in test_loader:
                if device == 'cpu':
                    start = time.time()
                else: #cuda
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()  #cuda
                data = image.to(device)
                prd = model.forward(data)
                # prd = prd['out'].to('cpu').detach()

                ## visualize result
                # print(prd['out'][0,:,:,:].shape)
                # overlayed_img = visualize_seg(image, prd)
                # print(overlayed_img.shape)
                # cv2.imshow('Result', overlayed_img)
                # cv2.waitKey(200)

                ## count latency
                if device == 'cpu':
                    duration = (time.time() - start) *1000 # metrics in ms
                else:  # gpu
                    ender.record()
                    torch.cuda.synchronize()  ###
                    duration = starter.elapsed_time(ender) # metrics in ms
                latency_list.append(duration)
                count += 1

                ## measure accuracy and record loss
                try:  # if train end, then terminate infer here
                    # if con_inf_b.poll() and config['verbose'] != True:  # verbose to control if this program avoke locally (cnn_infer.py)
                    if con_b.poll() :
                        msg = con_b.recv()  # receive msg from train
                        if msg == 'stop':  # signal from training process to end, ROy added on 12092022
                            print('Infer: get "stop" notice from main. ')
                            con_b.close()
                            latency = np.mean(latency_list[3:])  # take off GPU warmup
                            print('Infer: Quiting..., average latency is %0.5f ms. Total %d instances are infered.'% (latency,count*batch_size))
                            queue.put(dict(latency_ms=latency))  # queue.put(dict(process = 'infer')), {'latency': latency}
                            quit_while = True   # so, it will quit both for loop and while loop

                            break
                except Exception as e:
                    print(e)
                    break

            if quit_while == True:
                break
            ### ^^^^^^Above is for stoping infer after receiving msg from cnn_train.py^^^^^^^^^^ ####

    print('Inferenc latency is: ', latency/batch_size, 'ms. Total ', count*batch_size,' images are infered.' )

if __name__ == '__main__':
    queue = mp.Queue()
    pipe3, pipe4 = mp.Pipe()
    pipe =(pipe3, pipe4)
    config = {"arch": "deeplabv3_resnet101", "workers": 1, "epochs": 3, "batch_size":8,  "image_size": 224,  "device": "cuda"}
    work_train(config, queue)

    # config = {"arch": "deeplabv3_resnet101", "workers": 1, "epochs": 3, "batch_size": 1, "image_size": 224, "device": "cuda"}
    # work_infer(config, pipe, queue)


