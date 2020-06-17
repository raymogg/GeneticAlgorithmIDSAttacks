from GeneticAlgorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
from labels import data_headers
import numpy as np

# File for testing the algorithm
# Includes functions for hyperparameter testing, computing statistics of tests
# and plotting


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

def plotTestingData(testingData):
    x = []
    y = []
    for entry in testingData.keys():
        x.append(entry)
        y.append(testingData[entry][0][2])

    print(x)
    print(y)
    plt.plot(x, y)
    plt.show()

#Runs a basic test on the algorithm and plots the returned sample
def basicAlgorithmTest():
    algorithm = GeneticAlgorithm(False, 18, "nmap", True)
    model = algorithm.getModel()
    final_population = algorithm.run_algorithm(20, 120, 30)
    print("Algorithm Execution Finished")
    
    #Show the final population statistics
    statistics = calculatePopulationStatistics(final_population)
    print(" ")
    print("Run Statistics")
    print("Most Fit Sample Fitness: " + str(statistics[0]))
    print("Least Fit Sample Fitness: " + str(statistics[1]))
    print("Number of Attack Samples: " + str(statistics[2]))

    #From the final population, only pick samples that are NOT attacks
    only_benign = []
    for sample in final_population:
        if (sample['attack'][0] == 0):
            only_benign.append(sample)

    print("Most Fit Sample")
    print(only_benign[len(only_benign) - 1]['sample'])

    #Plot model used to evaluate samples
    #Plot original sample and fittest final sample
    fig, ax = plt.subplots()
    x_range = np.arange(38)
    width = 0.2
    seed_sample = ax.bar(x_range, extractPlotData(algorithm.getSeedAttack()), width, color='r')
    #For demo --> remove src bytes if needed to view sample in more detail
    only_benign[len(only_benign) - 1]['sample'][4] = 0
    only_benign[len(only_benign) - 1]['sample'][5] = 0
    fittest_sample = ax.bar(x_range+width, extractPlotData(only_benign[len(only_benign) - 1]['sample']), width, color='y')
    
    # seed_sample = ax.bar(x_range, extractPlotData([0,'udp','private' ,'SF' ,28 ,0 ,0 ,3, 0 ,0 ,0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 10, 10, 0.0, 0.0,0.0, 0.0, 1.0, 0.0, 0.0, 35, 10, 0.29, 0.11, 0.29, 0.0, 0.0, 0.0, 0.0, 0.0, 'teardrop',11]), width, color='r')
    # fittest_sample = ax.bar(x_range+width, extractPlotData([0, 'udp', 'private', 'SF', 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 10, 2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 35, 10, 0.29, 0, 0.29, 0.0, 0, 0.0, 0.0, 0, 'teardrop', 11]), width, color='y')
    ax.legend((seed_sample[0], fittest_sample[0]), ('Parent', 'Attack'))
    plt.xticks(labels=extractPlotLabels(data_headers), ticks=range(38), rotation=90)
    plt.grid(True)
    plt.show()



#Testing mutation variable
#Runs through a range of mutation variable values and computes fitness scores
#and number of attack vs benign samples
def testMutationVariable():
    #Test each variable 5 times
    data = {}
    for i in range(0, 55, 5):
        data[i] =  []
        algorithm = GeneticAlgorithm(False, i, "nmap", False)
        for j in range(5):
            result = algorithm.run_algorithm(algorithm.getSeedAttack(), 20, 20, 10)
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

#Function for testing the number of generations produced
def testGenerationsVariables():
    #Testing number of generations vs fitness
    data = {}
    algorithm = GeneticAlgorithm(False, 20, "teardrop", False)
    for i in range(200, 550, 50):
        data[i] = []
        result = algorithm.run_algorithm(algorithm.getSeedAttack(), i, 20, 10)
        result_metrics = calculatePopulationStatistics(result)
        print(result_metrics)
        data[i].append(result_metrics)

#Testing number of number of offspring and fittest num variables (number of fittest off spring to take)
#Tests the ratio of offspring to fittest numbers that produce the best results.
def testPopulationVariables():
    data = {}
    algorithm = GeneticAlgorithm(False, 20, "teardrop", False)
    for i in range(2, 10, 1):
        data[i] = []
        result = algorithm.run_algorithm(20, 30 * i, 30)
        result_metrics = calculatePopulationStatistics(result)
        print(result_metrics)
        data[i].append(result_metrics)

    print(data)
    plotTestingData(data)

basicAlgorithmTest()
