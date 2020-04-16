from GeneticAlgorithm import GeneticAlgorithm
import matplotlib.pyplot as plt

def calculatePopulationStatistics(population):
        #Fittest sample, weakest sample, number of attacks
        num_attacks = 0
        for sample in population:
            if (sample['attack']):
                num_attacks += 1
        return [population[-1]['fitness'], population[0]['fitness'], num_attacks]

def extractPlotData(sample):
    plot_data = []
    for i in range(len(nmap_attack)):
        if (i == 1 or i == 2 or i == 3 or i == 41 or i == 42):
            continue
        else:
            plot_data.append(teardrop_attack[i])
    
    return plot_data

nmap_attack = [0,'icmp','eco_i','SF',8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,15,0.00,0.00,0.00,0.00,1.00,0.00,1.00,2,46,1.00,0.00,1.00,0.26,0.00,0.00,0.00,0.00,'nmap',17]
teardrop_attack = [0, 'udp', 'private', 'SF', 28, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 82, 82, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 255, 182, 0.71, 0.29, 0.71, 0.00, 0.09, 0.00, 0.20, 0.00,'teardrop',16]

#Basic run of algorithm Begin Algorithm
# algorithm = GeneticAlgorithm(True, 20, "teardrop")
# print("Starting Sample")
# print(teardrop_attack)
# model = algorithm.getModel()
# final_population = algorithm.run_algorithm(teardrop_attack, 2, 4, 2)

# #Plot original sample and fittest final sample
# plt.bar(x=range(38), height=extractPlotData(teardrop_attack))
# plt.show()


#Testing mutation variable
#Test each variable 5 times

data = {}
for i in range(0, 55, 5):
    data[i] =  []
    algorithm = GeneticAlgorithm(False, i, "nmap")
    for j in range(5):
        result = algorithm.run_algorithm(nmap_attack, 20, 20, 10)
        result_metrics = calculatePopulationStatistics(result)
        print(result_metrics)
        data[i].append(result_metrics)

avg_results = {}
for k in data.keys():
    sum_max = 0
    sum_min = 0
    sum_attacks = 0
    for row in data[k]:
        sum_max += row[0]
        sum_min += row[1]
        sum_attacks += row[2]
    
    avg_max = sum_max / 5.0
    avg_min = sum_min / 5.0
    avg_attacks = sum_attacks / 5
    avgs = [avg_max, avg_min, avg_attacks]
    avg_results[k] = avgs
print(avg_results)

