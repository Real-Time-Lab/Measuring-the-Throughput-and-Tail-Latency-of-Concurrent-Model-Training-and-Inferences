'''
Transfer learning demo, 2 binary classification of bee and ants

Base model: cnn_models such as vgg, resnet, mobilenet, etc
TL model (change the final layer to a binary classification

dataset: download from: https://download.pytorch.org/tutorial/hymenoptera_data.zip
extract to os.path.join(root, './Documents/hymenoptera_data')

'''

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms, utils
import time
import os
import copy
import multiprocessing as mp
from tqdm import tqdm
import argparse
import cnn_train  #for accuracy , TBI
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append("../src")
from caltech101 import GenerateDataset

parser =  argparse.ArgumentParser()
parser.add_argument('--dataset', metavar = 'dataset', default= './Documents/caltech101', help ='name of dataset: e.g. caltech101, or, hymenoptera_data ')
parser.add_argument('--arch', metavar = 'arch', default = 'alexnet', help ='e.g. resnet50, alexnet')
parser.add_argument('--method', metavar = 'method', default = 'append', help ='e.g. fine-tune or append') # train as fine tune or append
args = parser.parse_args()
root = os.environ['HOME']

cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']



def work_train(config, queue): ## method: fine tune or append

    for key, value in config.items():  # update the args with transfered config
        vars(args)[key] = value

    device = args.device
    batch_size = args.batch_size
    image_size = args.image_size
    num_workers = args.workers
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model_name = args.arch
    assert model_name in cnn_model_list, f'Model \"{args.arch}\" is not recognized!'
    model_func = 'models.' + model_name
    model = eval(model_func)(weights=True)  # eval(): transform string to variable or function, # weights: pre-trained = True

    print('Training of transfer learning starts...')
    print('Device:', device)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size + 32),  ## 224+32=256
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if args.dataset == 'caltech101':
        image_datasets = {x: GenerateDataset(image_size=(224, 224), mode=x) for x in ['train', 'val']}
        dataloaders  = {x: DataLoader(image_datasets[x], batch_size= batch_size, num_workers=1, shuffle=True) for x in ['train', 'val']}
        class_names = np.unique(image_datasets['train'].label_list_trn)
    else:
        data_dir = os.path.join(root, './Documents/datasets/hymenoptera_data')
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= batch_size,
                                                      shuffle=True, num_workers= num_workers)
                       for x in ['train', 'val']}

        class_names = image_datasets['train'].classes
        # inputs, classes = next(iter(dataloaders['train']))
        # out = utils.make_grid(inputs)

    if args.method == 'append':
        ## option 1 : train final layers
        model_conv = model  # weights: pre-trained = True
        for param in model_conv.parameters():
            param.requires_grad = False  ## freeze train
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, len(class_names))  # num_class =2 , change the final layers
        model_conv = model_conv.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
        train_model(model_conv, args, criterion, optimizer_conv,
                    exp_lr_scheduler, dataloaders=dataloaders)
    else: #method == 'fine-tune'
        ## option 2 train the whole fine-tune model
        model_ft = model
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))  # num_class =2
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        train_model(model_ft, args, criterion, optimizer_ft, exp_lr_scheduler,
                               dataloaders= dataloaders )

def work_infer(config, pipe, queue):
    for key, value in config.items():
        vars(args)[key] = value  # update args with config passed
    print(vars(args))

    con_tl_a,con_tl_b = pipe
    con_tl_a.close()

    queue.put(dict(process= 'deeplab_infer'))  # for queue exception of this process
    model_name= args.arch
    batch_size = int(args.batch_size)
    img_size = int(args.image_size)
    device = args.device
    dataset = args.dataset
    num_workers = args.workers

    model_name = args.arch
    assert model_name in cnn_model_list, f'Model \"{args.arch}\" is not recognized!'

    data_transforms =   transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    if args.dataset == 'caltech101':
        image_datasets = GenerateDataset(image_size=(224, 224), mode='test')
        dataload_test  = DataLoader(image_datasets, batch_size= batch_size, num_workers=1, shuffle=True)

    else:
        data_dir = os.path.join(root, './Documents/datasets/hymenoptera_data')
        image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms)  # no transforms here
        dataload_test = torch.utils.data.DataLoader(image_datasets, batch_size= batch_size,
                                                      shuffle=True, num_workers= num_workers)
    pt_dir = os.path.join(root, './Documents/pt_files')
    pt_file_name = args.arch + '_tfl_best.pt'
    pt_file_path = os.path.join(pt_dir, pt_file_name)
    assert os.path.exists(pt_file_path), 'This model by transfer learning is not exist, please train it first.'
    model = torch.load(pt_file_path)  # /home/royliu/Documents/pt_files/resnet50_best.pt

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
            for data, label in dataload_test:
                if device == 'cpu':
                    start = time.time()
                else:  # cuda
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()  # cuda
                data = data.to(device)
                prd = model.forward(data)
                prd = prd.to('cpu').detach()
                # prd= prd.numpy()
                # prd = np.argmax(prd, axis=1)  # output is class_value

                ## count latency
                if device == 'cpu':
                    duration = (time.time() - start) * 1000  # metrics in ms
                else:  # gpu
                    ender.record()
                    torch.cuda.synchronize()  ###
                    duration = starter.elapsed_time(ender)  # metrics in ms
                latency_list.append(duration)
                count += 1

                target = label
                # measure accuracy and record loss
                acc1, _ = cnn_train.accuracy(prd, target, topk=(1, 2))
                if config['verbose'] == True:
                    print('acc: ', acc1.numpy()[0] / 100)  # convert tensor to numpy

                try:  # if train end, then terminate infer here
                    # if con_inf_b.poll() and config['verbose'] != True:  # verbose to control if this program avoke locally (cnn_infer.py)
                    if con_tl_b.poll():
                        msg = con_tl_b.recv()  # receive msg from train
                        if msg == 'stop':  # signal from training process to end, ROy added on 12092022
                            print('Infer: get "stop" notice from main. ')
                            con_tl_b.close()
                            latency = np.mean(latency_list[3:])  # take off GPU warmup
                            print('Infer: Quiting..., average latency is %0.5f ms. Total %d instances are infered.' % (
                            latency, count * batch_size))
                            queue.put(
                                dict(latency_ms=latency))  # queue.put(dict(process = 'infer')), {'latency': latency}
                            quit_while = True  # so, it will quit both for loop and while loop

                            break
                except Exception as e:
                    print(e)
                    break

            # if config['verbose'] == True:  # means the program is initiate her,i.e. cnn_infer.py, so quite after one epoch
            #     quit_while = True
            #     print('Infering done! Total: ', len(dataload_test), 'interations!')
            if quit_while == True:
                break
            ### ^^^^^^Above is for stoping infer after receiving msg from cnn_train.py^^^^^^^^^^ ####

    print('Inferenc latency is: ', latency / batch_size, 'ms. Total ', count * batch_size, ' images are infered.')
    # print('acc: ', acc1.numpy()[0] / 100)  # convert tensor to numpy

def train_model(model, args, criterion, optimizer, scheduler, dataloaders):
    num_epochs = args.epochs
    device = args.device

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            data_cnt = 0 # count number of processed instances for calculating lass and accuracy
            dl = dataloaders[phase]
            for inputs, labels in dl:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)  #inputs.size(0): batch size
                running_corrects += torch.sum(preds == labels.data)
                data_cnt += inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data_cnt
            epoch_acc = running_corrects.double() / data_cnt

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', end='', flush= True)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                ##save to file
                pt_dir = os.path.join(root, './Documents/pt_files')
                pt_file_name = args.arch + '_tfl_best.pt'
                pt_file_path = os.path.join(pt_dir, pt_file_name)
                if not os.path.exists(pt_dir): os.mkdir(pt_dir)
                torch.save(model, os.path.join(pt_dir,pt_file_path))  #/home/royliu/Documents/pt_files/resnet50_best.pt

    time_elapsed = time.time() - since
    time.sleep(0.5)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    queue.put(dict(duration_sec=time_elapsed))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    queue = mp.Queue()
    pipe_a, pipe_b = mp.Pipe()
    pipe =(pipe_a, pipe_b)
    config = {'arch': 'resnet50', 'workers': 1, 'epochs': 50, 'batch_size':1, 'image_size': 224, 'device': 'cuda',
              'verbose': True}
    # work_train(config, queue) # set batch size 1
    work_infer(config, pipe, queue)
#