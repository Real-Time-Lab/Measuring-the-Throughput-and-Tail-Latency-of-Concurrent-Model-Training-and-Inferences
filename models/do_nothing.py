def infer(config, pipe, queue):
    queue.put(dict(process= 'donothing_infer'))  # for queue exception of this process
    con_a,con_b = pipe
    con_a.close()
    print('Infer: do nothing')

    while True:
        pass
        if con_b.poll():
            msg = con_b.recv()  # receive msg from train
            if msg == 'stop':  # signal from training process to end, ROy added on 12092022
                print('Do noting: get "stop" notice from main. ')
                con_b.close()
                queue.put(dict(latency=0))
                print('Do noting: Quiting...')
                break

def train(config, queue):
    queue.put(dict(duration_sec= 0))
    print('Training: do nothing')

