'''
Function: scald down Azure trace

Original trace data: Traces downloaded from:https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md
Download it and put in apath (anywhere you like), here we put in:'./Documents/exps/profile-training-inference-cowork-73/traces/azurefunctions-dataset2019/invocations_per_function'

'''

import pandas as pd
import numpy as np
import os


path = os.path.join(os.environ['HOME'], './Documents/exps/profile-training-inference-cowork-73/traces/azurefunctions-dataset2019/invocations_per_function')
df = pd.read_csv(os.path.join(path,"./invocations_per_function_md.anon.d01.csv"))

# Store column names in a list
df_columns = df.columns.values.tolist()
df_columns.remove('HashOwner')
df_columns.remove('HashApp')
df_columns.remove('HashFunction')
df_columns.remove('Trigger')

# invocation_per_minute_per_day
invocations_per_minute = []

for i in range(len(df_columns)):
    invocations_per_minute.append(df[df_columns[i]].sum())

# requests per second
invocations_per_second = [int(x/60) for x in invocations_per_minute]
# print("Invocations per second:", invocations_per_second)

# Divide by 1000 to scale down
scaled_down_invocations_per_second = [int(x/1000) for x in invocations_per_second]
print("Scaled down invocations per second", scaled_down_invocations_per_second)

# To cover all 24 hours, choose requests per second for every 5th minute
request_rate_every_5th_minute = []
for i in range(len(scaled_down_invocations_per_second)):
        if i % 5 == 0:
            request_rate_every_5th_minute.append(scaled_down_invocations_per_second[i])

print("req every 5th minute length",len(request_rate_every_5th_minute))

# Interval between requests
delay_between_requests = [1/x for x in request_rate_every_5th_minute]
print("Delay between requests:",delay_between_requests)

delays_to_be_used_for_experiments = []

for i in range(len(request_rate_every_5th_minute)):
    for j in range(request_rate_every_5th_minute[i]):
        delays_to_be_used_for_experiments.append(delay_between_requests[i])

print(len(delays_to_be_used_for_experiments))

## save scaled trace file
df = pd.read_csv(os.path.join(path,"./invocations_per_function_md.anon.d01.csv"))
df = pd.DataFrame(delays_to_be_used_for_experiments)
df.to_csv(os.path.join('../result/delays_to_be_used_for_AZURE_TRACES_experiments-Day01.csv'), index=False, header=False)