from GeneticAlgorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
from labels import data_headers
import numpy as np

def calculatePopulationStatistics(population):
        #Fittest sample, weakest sample, number of attacks
        num_attacks = 0
        for sample in population:
            if (sample['attack']):
                num_attacks += 1
        return [population[-1]['fitness'], population[0]['fitness'], num_attacks]

def extractPlotData(sample):
    plot_data = []
    for i in range(len(sample)):
        if (i == 1 or i == 2 or i == 3 or i == 41 or i == 42):
            continue
        else:
            plot_data.append(sample[i])
    
    return plot_data

def extractPlotLabels(rawLabels):
    plot_labels = []
    for i in range(len(rawLabels)):
        if (i == 1 or i == 2 or i == 3 or i == 41 or i == 42):
            continue
        else:
            plot_labels.append(rawLabels[i])
    
    return plot_labels

nmap_attack = [0,'icmp','eco_i','SF',8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,15,0.00,0.00,0.00,0.00,1.00,0.00,1.00,2,46,1.00,0.00,1.00,0.26,0.00,0.00,0.00,0.00,'nmap',17]
teardrop_attack = [0, 'udp', 'private', 'SF', 28, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 82, 82, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 255, 182, 0.71, 0.29, 0.71, 0.00, 0.09, 0.00, 0.20, 0.00,'teardrop',16]

#Below is various testing functions. TODO Make this a class and pick what to run

#Basic run of algorithm Begin Algorithm
# algorithm = GeneticAlgorithm(True, 20, "teardrop")
# print("Starting Sample")
# print(teardrop_attack)
# model = algorithm.getModel()
# final_population = algorithm.run_algorithm(teardrop_attack, 2, 4, 2)



#Testing mutation variable
#Test each variable 5 times
# data = {}
# for i in range(0, 55, 5):
#     data[i] =  []
#     algorithm = GeneticAlgorithm(False, i, "nmap")
#     for j in range(5):
#         result = algorithm.run_algorithm(nmap_attack, 20, 20, 10)
#         result_metrics = calculatePopulationStatistics(result)
#         print(result_metrics)
#         data[i].append(result_metrics)

# avg_results = {}
# for k in data.keys():
#     sum_max = 0
#     sum_min = 0
#     sum_attacks = 0
#     for row in data[k]:
#         sum_max += row[0]
#         sum_min += row[1]
#         sum_attacks += row[2]
    
#     avg_max = sum_max / 5.0
#     avg_min = sum_min / 5.0
#     avg_attacks = sum_attacks / 5
#     avgs = [avg_max, avg_min, avg_attacks]
#     avg_results[k] = avgs
# print(avg_results)

#Plotting of produced attacks
# algorithm = GeneticAlgorithm(False, 25, "teardrop")
# results = algorithm.run_algorithm(teardrop_attack, 20, 20, 10)
# print(results)
# # #Plot original sample and fittest final sample
# fig, ax = plt.subplots()
# x_range = np.arange(38)
# width = 0.2
# seed_sample = ax.bar(x_range, extractPlotData(teardrop_attack), width, color='r')
# fittest_sample = ax.bar(x_range+width, extractPlotData(results[len(results) - 1]['sample']), width, color='y')
# ax.legend((seed_sample[0], fittest_sample[0]), ('Parent', 'Attack'))
# plt.xticks(labels=extractPlotLabels(data_headers), ticks=range(38), rotation=90)
# plt.grid(True)
# plt.show()

##Testing number of generations vs fitness
data = {}
algorithm = GeneticAlgorithm(False, 20, "teardrop")
for i in range(200, 550, 50):
    data[i] = []
    result = algorithm.run_algorithm(teardrop_attack, i, 20, 10)
    result_metrics = calculatePopulationStatistics(result)
    print(result_metrics)
    data[i].append(result_metrics)

print(data)
