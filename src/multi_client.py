import client
import os
import mp_exception as mp_new
import itertools
import random

cnn_model_list = ['alexnet', 'convnext_base', 'densenet121', 'densenet201', 'efficientnet_v2_l', \
                  'googlenet', 'inception_v3', 'mnasnet0_5', 'mobilenet_v2', 'mobilenet_v3_small', \
                  'regnet_y_400mf', 'resnet18', 'resnet50', 'resnet152', 'shufflenet_v2_x1_0', \
                  'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg16', 'vgg19', 'vit_b_16']
yolo_model_list = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
deeplab_model_list = ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet_v3_large']

def image_folder(data_dir, model):
    if model in cnn_model_list:
        # folder = './mini_imagenet/images'
        folder = './coco/images/test2017'
    elif model in deeplab_model_list:
        folder = './google_street/part1'
    else:
        folder = './coco/images/test2017'
    return os.path.join(data_dir,folder)


def create_model_comb_list(arch_pool, client_num):
    combinations = itertools.combinations(arch_pool, client_num)
    combinations_list = []
    for tuple in combinations:
        combinations_list.append(list(tuple))
    ## Uncomment following for random sampling
    # random.seed(12)
    # combinations_list = random.sample(combinations_list, 8)

    return combinations_list

def main():
    '''
    pick up any 2/3/4 models from arch_pool (10), run concurrently
    '''

    ################################  set configurations below ! #####################################################
    arch_zoo = ['resnet50', 'vgg16', 'mobilenet_v3_small', 'inception_v3', 'efficientnet_v2_l', 'densenet121', 'googlenet', 'alexnet']
    combinations = itertools.combinations(arch_zoo, 4)
    combinations_list = []
    for tuple in combinations:
        combinations_list.append(list(tuple))
    # print("Randamly chosen 5 combinations: ", random_combinations)
    # print("Number of combinations: ", len(random_combinations))

    basic_ip ='127.0.0.1'  ## change to your real ip address

    basic_port = 54100
    # client_num = 4 # set the combination number here, which means how many madels run concurrently, e.g. 2, 3, 4
    print_interval = 1000
    ## train_model_list = ['none', 'resnet152_8', 'efficientnet_v2_l_2', mobilenet_v3_small_128]
    train_model_name = 'resnet50_16' ## add this manually from: ## train_model_list = ['none', 'resnet152_8', 'efficientnet_v2_l_2', mobilenet_v3_small_128]
    request_rate_list = [0]  ## in ms, [10, 20, 40, 60] 0 =AZURE Traces
    client_num_list = [4,3,2] ## [4,3,2] means deploy 4,3,2 combinations of concurrently running inference tasks
    #####################################set parameters above !#####################################################

    for client_num in client_num_list:
        assert client_num < 11, f'server num. {client_num} is too big, no larger than 11.'  # support 9 servers at most
        addr_list = []
        [addr_list.append((basic_ip, port)) for port in range(basic_port, basic_port + client_num)]
        model_comb_list = create_model_comb_list(arch_zoo, client_num)
        # data_dir = os.path.join(os.environ['HOME'], './Documents/datasets')

        ###########################################################################################################################
        # uncomment following to use all combinations and comment out the following line
        # model_comb_list = create_model_comb_list(arch_pool, client_num)

        # put failed model combinations in the follwing list and comment out the above line
        # model_comb_list = [['mobilenet_v3_small', 'inception_v3', 'googlenet', 'alexnet'], ['efficientnet_v2_l', 'densenet121', 'googlenet', 'alexnet'], ['inception_v3', 'efficientnet_v2_l', 'densenet121', 'googlenet'], ['inception_v3', 'densenet121', 'googlenet', 'alexnet'], ['inception_v3', 'efficientnet_v2_l', 'densenet121', 'alexnet'], ['mobilenet_v3_small', 'densenet121', 'googlenet', 'alexnet'], ['mobilenet_v3_small', 'efficientnet_v2_l', 'densenet121', 'googlenet'], ['mobilenet_v3_small', 'efficientnet_v2_l', 'googlenet', 'alexnet'], ['inception_v3', 'efficientnet_v2_l', 'googlenet', 'alexnet'], ['mobilenet_v3_small', 'efficientnet_v2_l', 'densenet121', 'alexnet']]
        ############################################################################################################################

        for model_comb in model_comb_list: ## model_comb is a list of models
            p_list = []
            comb_id = 'models_' + str(client_num)+ "_"  # don't contain ':' or '|'
            if len(model_comb) >= 2:  # if there is multiple model inference run concurently, create subfolder to store result
                for model in model_comb : comb_id = comb_id + model + '+'
                comb_id = comb_id[:-1]
            print(comb_id)
            for i, addr in enumerate(addr_list):
                print(f'{model_comb[i]} is runing')
                ip, port = addr
                try:
                    ### call: work(ip, port, request_rate_list, arch_list, train_model_name, print_interval )
                    config= (ip, port, comb_id, request_rate_list , [model_comb[i]],train_model_name, print_interval)
                    p_list.append(mp_new.Process(target=client.work, args=(config)))
                except Exception as e:
                    print(e)
                    break

            for p in p_list: p.start()
            for p in p_list: p.join()
            for p in p_list: p.terminate()

if __name__ == '__main__':
    main()
