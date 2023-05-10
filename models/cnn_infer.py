import numpy as np
import sys
sys.path.append("../datasets")
# sys.path.append("..")
from mini_imageNet import MiniImageNet
from torch.utils.data import DataLoader
from torchvision import transforms, models
import time
import torch
import cnn_train
import multiprocessing as mp
import imageNet
import argparse

parser =  argparse.ArgumentParser()
parser.add_argument('--dataset', metavar = 'MiniImageNet', default= 'MiniImageNet', help =' to choose "MiniImageNet" or "imagenet" ')
parser.add_argument('--arch', metavar = 'arch', default = 'alexnet', help ='e.g. resnet50, alexnet')


def work(config, pipe, queue):
    args = parser.parse_args()
    for key, value in config.items():  # update the args with transfered config
        vars(args)[key] = value

    # root = args.root
    con_inf_a,con_inf_b = pipe
    con_inf_a.close()
    queue.put(dict(process= 'cnn_infer'))  # for queue exception of this process
    model_name= args.arch
    batch_size = args.batch_size
    img_size = args.image_size
    device = args.device
    dataset = args.dataset
    num_workers = args.workers

    if not 'verbose' in config:
        config['verbose'] = None  ## this program is avoked by main_.py

    transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop((img_size,img_size)),  #224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((img_size+32,img_size+32)), #256
                                   transforms.CenterCrop((img_size,img_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize((img_size+32,img_size+32)),
                                    transforms.CenterCrop((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }


    assert dataset in ['MiniImageNet', 'imagenet'], f'Dataset \"{dataset}\" is not recognized!'
    model_func = 'models.'+ model_name
    if dataset == 'MiniImageNet':
        data_test = MiniImageNet('test', transform)
        dataload_test = DataLoader(data_test, batch_size= batch_size, shuffle=True, num_workers=num_workers)
    elif dataset == 'imagenet':
        dataload_test = imageNet.test_loader(batch_size=batch_size, workers=num_workers, image_size = (img_size,img_size))

    model = eval(model_func)(pretrained=True) # eval(): transform string to variable or function
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
            for data, label in dataload_test:
                if device == 'cpu':
                    start = time.time()
                else: #cuda
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()  #cuda
                data = data.to(device)
                prd = model.forward(data)
                prd = prd.to('cpu').detach()

                ## print result
                # result= prd.numpy()
                # result = np.argmax(result, axis=1)  # output is class_value
                # print(result)

                ## count latency
                if device == 'cpu':
                    duration = (time.time() - start) *1000 # metrics in ms
                else:  # gpu
                    ender.record()
                    torch.cuda.synchronize()  ###
                    duration = starter.elapsed_time(ender) # metrics in ms
                latency_list.append(duration)
                count += 1

                target = label
                # measure accuracy and record loss
                acc1, acc5 = cnn_train.accuracy(prd, target, topk=(1, 5))
                if config['verbose'] == True:
                    print('acc1: ', acc1.numpy()[0]/100, ',        acc5: ', acc5.numpy()[0]/100)  # convert tensor to numpy


                try:  # if train end, then terminate infer here
                    # if con_inf_b.poll() and config['verbose'] != True:  # verbose to control if this program avoke locally (cnn_infer.py)
                    if con_inf_b.poll() :
                        msg = con_inf_b.recv()  # receive msg from train
                        if msg == 'stop':  # signal from training process to end, ROy added on 12092022
                            print('Infer: get "stop" notice from main. ')
                            con_inf_b.close()
                            latency = np.mean(latency_list[3:])  # take off GPU warmup
                            print('Infer: Quiting..., average latency is %0.5f ms. Total %d instances are infered.'% (latency,count*batch_size))
                            queue.put(dict(latency_ms=latency))  # queue.put(dict(process = 'infer')), {'latency': latency}
                            quit_while = True   # so, it will quit both for loop and while loop

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

    print('Inferenc latency is: ', latency/batch_size, 'ms. Total ', count*batch_size,' images are infered.' )
    print('acc1: ', acc1.numpy()[0]/100, ',        acc5: ', acc5.numpy()[0]/100)  # convert tensor to numpy


if __name__ == '__main__':

    config ={'arch': 'densenet201','workers': 1, 'epochs': 10, 'batch_size': 32, 'image_size':224, 'device':'cuda', 'verbose': True}
    pipe3, pipe4 = mp.Pipe()
    queue = mp.Queue()
    pipe =(pipe3, pipe4)
    ## note:  loop of inference start from here is infinete since no cnn_train.py stop it
    ## set verbose true here, otherwise Pipe will make the infer only take once (of course, I can add one more variable to control, but I didn't want).
    work(config, pipe, queue)

