from labels import data_headers
from processing import model
import pandas as pd

#ATTACK = [0,'icmp','eco_i','SF',8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,15,0.00,0.00,0.00,0.00,1.00,0.00,1.00,2,46,1.00,0.00,1.00,0.26,0.00,0.00,0.00,0.00,'nmap',17]
ATTACK = [0,'udp','private','SF',28,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,82,82,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,182,0.71,0.29,0.71,0.00,0.09,0.00,0.20,0.00,'teardrop',16]

def deviation(sample):
    std = 7116454.774411983
    total_deviation = 0
    for idx, val in enumerate(sample):
        #Only measure deviation on int values
        if (type(val) is int or type(val) is float):
            this_deviation = abs(val - ATTACK[idx])
            total_deviation += this_deviation

    #Deviaiton scaling. This is to get deviation between 0 and 100. Done using experiments.
    total_deviation = total_deviation * (100.00/ (3 * std))
    #total_deviation = min(total_deviation, 100)
    return total_deviation

train_df = pd.read_csv("Datasets/NSL-KDD/KDDTrain+.txt", header=None, names=data_headers)
#dropped = train_df.drop(["unknown", "attack", "protocol_type", "service", "flag"], axis=1)
deviation_samples = train_df.apply (deviation, axis=1)
print(deviation_samples)
print(deviation_samples.mean())
print(deviation_samples.std())
print(deviation_samples.max())



