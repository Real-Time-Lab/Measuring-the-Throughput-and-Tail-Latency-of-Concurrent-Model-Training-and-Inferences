'''
Function:
co-run training and infering
profiler recoding data when train and infer

use Pipe , after training ending,  cnn_train.py send message to profiller.py and cnn_infer.py to stop.

note: mp_exception.py is process cascade exception from children process to parent process
        https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent


inference latency prediction when co-run.

e.g. cnn config:
[{'arch': 'alexnet', 'workers': 1, 'batch_size': 1, 'image_size': 224, 'device': 'cuda'}, {'arch': 'mobilenet_v2', 'workers': 1, 'batch_size': 1, 'image_size': 224, 'device': 'cuda'}, {'arch': 'resnet152', 'workers': 1, 'batch_size': 1, 'image_size': 224, 'device': 'cuda'}]

'''

import json
import os
import multiprocessing as mp
import pandas as pd
import time
import mp_exception as mp_new  # procss.py is process cascade exception from children process to parent process
import traceback
import profiler
import sys
sys.path.append('../models')
sys.path.append('../datasets')
import cnn_infer, cnn_train, deeplab_v3, yolo_v5, do_nothing
import warnings
warnings.filterwarnings("ignore")

cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large']
model_list = cnn_model_list + yolo_model_list + deeplab_model_list

def date_time():
    s_l = time.localtime(time.time())
    dt = time.strftime("%Y%m%d", s_l)
    tm = time.strftime("%H%M%S", s_l)
    # print(date, tm )
    return dt, tm

def main():

    file = './config.json'
    dt, tm = date_time()
    log_file_name = 'profiler_log_' + dt + tm + '.csv'

    with open(file) as f:
        config = json.load(f)
    train_config_list, infer_config_list= config['train'],  config['infer']  #length: train: 36, infer: 12
    if not infer_config_list: infer_config_list = [{'arch': 'None'}]
    i=0
    profile_log = pd.DataFrame(columns = ['time_frame', 'train_configure', 'infer_configure','status', 'result'], index=None)

    profiling_num = 200000  #  every 5 epochs with imagenet: 5*1000*60 sec, 83 hours
    ## start co-run train and infer.....
    for infer_config in infer_config_list:
        for train_config in train_config_list:
            time.sleep(6)  # sleep 60 seconds among every combination experiment
            print('\n' * 5)
            print('======== round ', i, ': ==========')
            i += 1
            config, config_tr, config_inf = '', '', ''  #used as the configuration content and file name
            con_prf_a, con_prf_b = mp.Pipe()  # con_b in profiler message to main when complete task
            con_inf_a, con_inf_b = mp.Pipe()
            # con_prf_a.close()
            # con_inf_b.close()  # don't close it , otherwise pipe will break
            trn_queue = mp.Queue()
            inf_queue = mp.Queue()

            try:  # some models in the list main not available, so use it to bypass
                for key, value in train_config.items(): config_tr += (' ' + key + '_'+ str(value))
                for key, value in infer_config.items(): config_inf += (' ' + key + '_'+str(value))
                config = 'Train'+config_tr +' + '+ 'Infer'+ config_inf
                print(config)
                p1 = mp.Process(target= profiler.profile, args = (config, profiling_num, (con_prf_a,con_prf_b),))

                if train_config['arch'] in deeplab_model_list:
                    p2 = mp_new.Process(target=deeplab_v3.work_train, args=(train_config,),
                                        kwargs=(dict(queue=trn_queue)))
                elif train_config['arch'].lower() == 'none': # no training
                    p2= mp_new.Process(target=do_nothing.train, args=(train_config,),
                                        kwargs=(dict(queue=trn_queue)))
                else:
                    p2 = mp_new.Process(target= cnn_train.work, args= (train_config,), kwargs=(dict(queue=trn_queue)))

                if infer_config['arch'] in yolo_model_list:
                    p3 = mp_new.Process(target = yolo_v5.work_infer, args = (infer_config, (con_inf_a,con_inf_b),), kwargs=(dict(queue=inf_queue)))
                elif infer_config['arch'] in deeplab_model_list:
                    p3 = mp_new.Process(target = deeplab_v3.work_infer, args = (infer_config, (con_inf_a,con_inf_b),), kwargs=(dict(queue=inf_queue)))
                elif infer_config['arch'] in cnn_model_list: # in cnn list
                    p3 = mp_new.Process(target = cnn_infer.work, args = (infer_config, (con_inf_a,con_inf_b),), kwargs=(dict(queue=inf_queue)))
                elif infer_config['arch'].lower() == 'none': #  infer_config['arch'] == 'None': # no inference
                    p3 = mp_new.Process(target=do_nothing.infer, args=(infer_config, (con_inf_a, con_inf_b),),
                                        kwargs=(dict(queue=inf_queue)))
                else:  # e.g. '2 inferences', manually run 2 inferences, but run train as 'none'combination
                    p3 = mp_new.Process(target=do_nothing.infer, args=(infer_config, (con_inf_a, con_inf_b),),
                                        kwargs=(dict(queue=inf_queue)))

                p_list=[p1, p2, p3]
                res = 'na'

                for p in p_list: p.start()

                while p2.is_alive() or p3.is_alive():

                    # below to break when profiler complete job
                    if con_prf_b.poll():
                        msg= con_prf_b.recv()
                        con_prf_b.close()
                        if msg == 'done':
                            print('Main: get "done" notice from profiler! ')
                            con_inf_a.send('stop')  # notice infer to stop
                            con_inf_a.close()

                            while not inf_queue.empty():  # take the last queue data (latency)
                                inf_queue.get()  # take the rest until last one
                            res = inf_queue.get()
                            print('Main: Get data from infer:', res)

                        print('Main: Terminate training and inference', '\n')
                        profile_log.to_csv(os.path.join('../result/log/all_profiler_log', log_file_name), index_label=None, mode='w')
                        print('log file saved in:', os.path.join('../result/log/all_profiler_log', log_file_name))
                        p2.terminate()
                        p3.terminate()
                        break

                    ## to break when train complete
                    if not trn_queue.empty():   ## training complete with setting epochs
                        res = trn_queue.get()  # achive training complete notice
                        if res['duration_sec'] >0 :  ## training time elapse
                            print('Main: get train done notice! ')
                            con_inf_a.send('stop')  # notice infer to stop
                            con_inf_a.close()
                            con_prf_a.send('stop')
                            con_prf_a.close()

                            print('Main: Terminate profiling, training and inference')
                            # profile_log.to_csv(os.path.join('../result/log/all_profiler_log', log_file_name), index_label=None, mode='w')
                            # print('log file saved in:', os.path.join('../result/log/all_profiler_log', log_file_name))
                            # p1.terminate()
                            # p2.terminate()
                            p3.terminate()

                            break

                    ######## below codes for receiving exception from subprocesser #########
                    if p2.exception:
                        print('Terminate all processes due to train process exception. ')
                        error, trc_back = p2.exception
                        p3.terminate()
                        p2.terminate()
                        p1.terminate()
                        raise ChildProcessError(trc_back)
                        profile_log.to_csv(os.path.join('../result/log/all_profiler_log', log_file_name), index_label=None, mode='w')
                        print('log file saved in:', os.path.join('../result/log/all_profiler_log', log_file_name))
                        break
                    if p3.exception:
                        print('Terminate all processes, due to infer process exception.')
                        error, trc_back = p3.exception
                        p3.terminate()
                        p2.terminate()
                        p1.terminate()
                        raise ChildProcessError(trc_back)
                        profile_log.to_csv(os.path.join('../result/log/all_profiler_log', log_file_name), index_label=None, mode='w')
                        print('log file saved in:', os.path.join('../result/log/all_profiler_log', log_file_name))
                        break

                for p in p_list:
                    p.join()

                for p in p_list:
                    p.terminate()

                profile_log.loc[i]= [time.time(), train_config, infer_config, 'Sucess',res]  # add one row of data
                profile_log.to_csv(os.path.join('../result/log/all_profiler_log', log_file_name), index_label=None, mode='w')

            except Exception:
                print('traceback:', traceback.format_exc())
                profile_log.loc[i] = [time.time(), train_config, infer_config, 'Fail', trc_back]
                profile_log.to_csv(os.path.join('../result/log/all_profiler_log', log_file_name), index_label=None, mode='w')

if __name__ == '__main__':
    main()