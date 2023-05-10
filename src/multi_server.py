import time
import server
# import multiprocessing as mp
import mp_exception as mp_new
import warnings
warnings.filterwarnings("ignore")

def main():
    # basic_ip ='127.0.0.1'   ## change to your real ip address
    basic_ip = '0.0.0.0'

    basic_port = 54100
    server_num = 4
    assert server_num < 11, f'server num. {server_num} is too big, no larger than 11.'  # support 9 servers at most
    addr_list = []
    [addr_list.append((basic_ip, port)) for port in range(basic_port ,basic_port +server_num)]
    # pool = mp.Pool(server_num)

    p_list=[]
    for i, addr in enumerate(addr_list):
        try:
            # pool.apply_async(server.work, args=(addr_list[i]),)  # change to below line for passing exception
            p_list.append(mp_new.Process(target = server.work, args=(addr_list[i])))

        except:
            break

    for p in p_list: p.start()
    for p in p_list: p.join()
    for p in p_list: p.terminate()

    # pool.close()
    # pool.join()

if __name__ == "__main__":
    main()