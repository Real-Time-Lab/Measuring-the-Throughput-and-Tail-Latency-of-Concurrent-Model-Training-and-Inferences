import json
import os
import re
import subprocess
import pandas as pd
import time
from util import save_log
import multiprocessing as mp
def cpu_freq():
    dev=json.load(open(r'./device.json','r'))
    cpu_freq_dict ={}
    for i in range(os.cpu_count()):

        cpu_name = 'cpu'+str(i)
        arg = dev['cpu'][cpu_name]['freq_now']
        cpu_freq_dict[cpu_name+'_freq']= subprocess.check_output(['cat', arg]).decode('utf-8').strip()
    return cpu_freq_dict


def cpu_usage():
    p = subprocess.Popen(["top", "n1", "b"], \
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    threads_info = stdout.decode('utf-8').split('\n')
    cpu_usg=threads_info[2].split(',')
    cpu_usg_val= re.findall(r'[0-9]+[.][0-9]+', str(cpu_usg))  # parse value
    cpu_usg_col = re.findall('[a-zA-Z]+', str(cpu_usg))[2:] # [:-2]: take off string of '%Cpu(s)'
#     cpu_usg = [float(value.strip("%"))/float(cpu_usg[0].strip("%"))*100 for value in cpu_usg]
#     thrd_num= len(threads_info) -5
#     cpu_usg_col.append('thrd_num')
#     cpu_usg_val.append(thrd_num)
    cpu_usg_dict = dict(zip(cpu_usg_col, cpu_usg_val))

    return cpu_usg_dict

def date_time():
    s_l = time.localtime(time.time())
    dt = time.strftime("%Y%m%d", s_l)
    tm = time.strftime("%H%M%S", s_l)
    # print(date, tm )
    return dt, tm

def fmt_to_stmp(tm,*arg):
    fmt= "%Y-%m-%d %H:%M:%S" if not arg else arg[0]
    dt, ms =tm.split('.')  # to process with milli-second
    timeArray =time.strptime(dt, fmt)
    timeStamp = time.mktime(timeArray)+int(ms)/1000
    return timeStamp

def grab_gpu_data(arg):
    query_dic = {'time_stamp':'timestamp', 'gpu_name':'name', 'index':'index','gpu_power':'power.draw',\
                 'gpu_freq':'clocks.gr', 'gpu_mem_freq':'clocks.mem','gpu_temp':'temperature.gpu',\
                 'gpu_util%':'utilization.gpu', 'gpu_mem_util%':'utilization.memory',\
                 'gpu_mem_total':'memory.total', 'gpu_mem_used':'memory.used'}

    pos, i = -1, 0 # the position of time_stamp for converting
    query = ''  #create an empty string
    for item in arg:
        if item == 'time_stamp':
            pos = i
        i+=1
        query+=query_dic[item]+','

    query='--query-gpu='+query[:-1]  # add query string's head and cut tail, which tail is ','
    nvidia_smi = "nvidia-smi"
    p = subprocess.Popen([nvidia_smi, query, "--format=csv,noheader,nounits"], stdout=subprocess.PIPE) # close_fds= True
    stdout,  stderror = p.communicate()
    output = stdout.decode('UTF-8').strip()
    output = output.split(',')  #split the returned string and convert to list
    output[pos]= fmt_to_stmp(output[pos], "%Y/%m/%d %H:%M:%S") # convert date-time to stamp format, format in "%Y/%m/%d %H:%M:%S"
    return dict(zip(arg, output))

# def save_log(data,config):
#     dt, tm = date_time()
#     log_dir = '../result/log/' + dt
#     if not os.path.exists(log_dir): os.mkdir(log_dir)
#     log_file = config + '_'+ dt + tm
#     data.to_csv(os.path.join(log_dir,log_file)+ '.csv')

def profile(config, profiling_num, pipe):
    file_name = 'profile_' + config
    con_prf_a, con_prf_b = pipe
    # con_prf_b.close()  ##  only send message to main, 关闭收到端
    gpu_col = ['time_stamp', 'gpu_power', 'gpu_freq', 'gpu_mem_freq', 'gpu_temp', 'gpu_util%', 'gpu_mem_util%', 'gpu_name']
    gpu_dict = grab_gpu_data(gpu_col)
    cpu_usg_dict = cpu_usage()
    cpu_freq_dict = cpu_freq()

    cpu_col = ['timestamp'] + list(cpu_usage().keys()) + list(cpu_freq().keys())  # create the cpu, gpu's columns of dataframe
    cpu = pd.DataFrame(columns=cpu_col, index=None)
    gpu = pd.DataFrame(columns=gpu_col, index=None)

    col_select = ['gpu_power', 'gpu_freq', 'gpu_mem_freq', 'gpu_temp', 'gpu_util%', 'gpu_mem_util%', 'us', 'sy',
           'cpu0_freq', 'cpu5_freq']

    start = time.time()
    i = 0

    # while True:
    print('profiling_num:', profiling_num)
    delay, duration,  interval_target =0.0, 0.9, 1 # profile every 1 sec
    time_prev = time.time()
    for i in range(profiling_num):

        try:
            # cpu data collection
            cpu_usg_dict = cpu_usage()
            cpu_freq_dict = cpu_freq()
            val = [time.time()] + list(cpu_usg_dict.values()) + list(cpu_freq_dict.values())
            cpu.loc[i, cpu_col] = val

            # gpu data collection
            val = list(grab_gpu_data(gpu_col).values())
            gpu.loc[i, gpu_col] = val

            ##### Below codes to creat a delay to meet target profiling interval, e.g. 1 sec. #########
            time_now = time.time()
            duration = time_now - time_prev
            diff = delay+ interval_target-duration
            if diff > 0.01:  # only change delay when need , make sure delay won't be negative value
                delay= delay + (interval_target-duration)
            elif interval_target-duration < 0:
                delay = 0
            if delay > 0: time.sleep(delay)
            time_prev = time_now
            # print(time.time())
            ######## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^########

            i += 1
            print('\r', i, ' instances.', end='')

        except:  # press stop to break
            print()
            print('Profiler: Collecting done!')
            data = gpu.join(cpu)
            save_log(data, file_name)
            # print(cpu_usg_dict)
            break

        if con_prf_b.poll():
            msg = con_prf_b.recv()
            con_prf_b.close()
            if msg == 'stop':
                print()
                print('Profiler: get notice from main to stop!')
                data = gpu.join(cpu)
                save_log(data, file_name)
                # print(cpu_usg_dict)
                break

    print(f'Data shape of gpu:{gpu.shape}, and cpu: {cpu.shape}.')
    data = gpu.join(cpu)
    save_log(data, file_name)
    # save_log(gpu, 'gpu')
    # save_log(cpu, 'cpu')
    print('Profiler: Profiling is ending, notice to main!')
    con_prf_a.send('done')
    print('Profiler: Time elapsed: ', time.time() - start, 'sec.')
    return 0

if __name__ == '__main__':
    profiling_num = 800000
    con_prf_a, con_prf_b = mp.Pipe()
    profile('Train_none+Infer_none',profiling_num, (con_prf_a, con_prf_b))