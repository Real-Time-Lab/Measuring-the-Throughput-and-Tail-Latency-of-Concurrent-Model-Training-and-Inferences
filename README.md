# Corun: Concurrent Inference and Continuous Training at the Edge for Cost-Efficient AI-Based Mobile Image Sensing
## "src" folder:
 - main.py
   - implement single training and single inference tasks to run concurrently
   - set training and inference co-running configure in config.json
 - profiler.py
   - run it for profiling GPU and CPU state, at a 1 second interval
   - hardware is set in device.json
 - multi_client.py and multi_server.py
   - Implement single training and multiple inference tasks to run concurrently: 
 - multi_infer.py
   - implement multiple inference tasks
 - client.py, client_asysnc.py, server.py
   - act as sub-functions of above functions 
   - client.py spawn client in synchronized mode, while client_asysnc.py is in an asynchronized mode
## "datasets" folder:
 - dataset and dataloader generators 
## "models" folder:
 - build DNN models
## "result" folder:
 - save results
