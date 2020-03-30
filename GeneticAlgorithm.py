from labels import data_headers, attack_generation_labels
from processing import model
import pandas as pd
import random

#Global Variables
DEBUG = False

#Get our model
print("Initialising Model")
_model = model()

#ATTACKS - TODO CHANGE FROM HARDCODED TO RANDOMLY PICKED
nmap_attack = [0,'icmp','eco_i','SF',8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,15,0.00,0.00,0.00,0.00,1.00,0.00,1.00,2,46,1.00,0.00,1.00,0.26,0.00,0.00,0.00,0.00,'nmap',17]
teardrop_attack = [0,'udp','private','SF',28,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,82,82,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,182,0.71,0.29,0.71,0.00,0.09,0.00,0.20,0.00,'teardrop',16]
ATTACK = nmap_attack

def validId(idx):
    if (idx == 1 or idx == 2 or idx == 3 or idx == 41 or idx == 42):
        return False
    else:
        return True
#Function used to evaluate a sample on the model
def evaluate_sample(model, sample):
    sample_df = pd.DataFrame([sample], columns=data_headers)
    dropped = sample_df.drop(["unknown", "attack", "protocol_type", "service", "flag"], axis=1)
    #encoded_df = pd.get_dummies(dropped, columns=["protocol_type", "service", "flag"])
    pred = _model.predict(dropped)
    return (pred)

def deviation(initial_sample, sample):
    total_deviation = 0
    for idx, val in enumerate(sample):
        #Only measure deviation on int values
        if (type(val) is int and validId(idx)):
            #Factor in the value range for the value
            max_value = attack_generation_labels[data_headers[idx]][-1]
            this_deviation = (abs(val - initial_sample[idx]) / max_value)
            total_deviation += this_deviation
    return total_deviation

# #Fitness Function
def fitness(initial_sample, sample):
    attack_eval = evaluate_sample(_model, sample)[0]
    _deviation = deviation(initial_sample, sample)

    if (DEBUG):
        print("Sample: ")
        print(sample)
        print(attack_eval)
        print(_deviation)
    
    #Original Sample has a fitness of 1
    if (_deviation == 0):
        return 1
    
    if (attack_eval == 0):
        #Not an attack, reward small deviation
        return 0.9 * (1.0/_deviation)
    else:
        #Is an attack, reward small deviation but less so than attack
        return 0.1 * (1.0/_deviation)

def add_mutation(sample):
    for idx, val in enumerate(sample):
        if (not validId(idx)):
            #Skip protocol type, service, flag, attack and unknown
            if (DEBUG):
                print("Skipping")
            continue
        
        #Mutate each gene with 5% change
        rand = random.randint(0, 100)
        if (rand <= 5):
            if (DEBUG):
                print("Mutating")

            #Mutate by picking from a random index within allowable range
            max_range = attack_generation_labels[data_headers[idx]][-1]
            index = random.randint(0, max_range)
            new_value = attack_generation_labels[data_headers[idx]][index]

            if (DEBUG):
                print("Current Value: " + str(sample[idx]))
                print("New Value: " + str(new_value))
            sample[idx] = new_value  

    return sample 

def generate_offspring(sample1, sample2):
    offspring = [None] * len(sample1) #-3 as we actually r
    #37 labels, for each one take randomly from each parent
    for val in range(len(sample1)):
        if (val == 1 or val == 2 or val == 3):
            #For protocol type, service, and flag, we simply match the parents
            offspring[val] = sample1[val]
            continue

        #Take genes from each parent.
        rand = random.randint(0, 1)
        if (rand):
            #Take from sample 1
            offspring[val] = sample1[val]
        else:
            offspring[val] = sample2[val]

    #Add mutation to the offspring
    #ßreturn offspring
    return add_mutation(offspring)

def key_extractor(sample):
    return sample['fitness']

def sample_extractor(sample):
    return sample['sample']

def display_population_statistics(population):
    print("FITTEST SAMPLE: "  + str(population[-1]['fitness']))
    print("WEAKEST SAMPLE: "  + str(population[0]['fitness']))
    num_attacks = 0
    for sample in population:
        if (sample['attack']):
            num_attacks += 1
    print("NUMBER OF SAMPLE ATTACKS: " + str(num_attacks))
    print(" ")

def display_population(popluation):
    print("POPULATION")
    for sample in popluation:
        print(sample)

#Function to run the algorithm
#model = the model used to evaluate attack samples
#initial_parent = the initial genetic parent used to breed offspring
#iterations = the number of iterations / generations to breed
#offspring_number = the number of offspring to breed per generation
#fittest_num = the number of offspring to pick to move through to the next generation
def run_algorithm(model, initial_parent, iterations, offspring_number, fittest_num):
    #Breed initial population
    print("Breeding Initial Population")
    population = []
    for i in range(offspring_number):
            population.append(generate_offspring(initial_parent, initial_parent))

    print("Running Genetic Algorithm")
    for j in range(iterations):
        print("GENERATION: " + str(j))
        offspring = []
        #Breed each pair in the population
        index = 0
        while (index < (len(population) - 1)):
            offspring.append(generate_offspring(population[index], population[index + 1]))
            index += 2
        
        #Place offspring in population
        population.extend(offspring)

        #Evaluate the fittest_num samples to go through to next population
        fittest_samples = []
        for sample in population:
            sample_fitness = fitness(initial_parent, sample)
            is_attack = evaluate_sample(model, sample)
            fittest_samples.append({'fitness': sample_fitness, 'sample': sample, 'attack': is_attack})
        fittest_samples.sort(key=key_extractor)
        
        #Trim population if too large
        if (len(fittest_samples) > fittest_num):
            population = fittest_samples[len(fittest_samples) - fittest_num: ]
        
        display_population_statistics(population)
        population = list(map(sample_extractor, population))

    display_population(population)
        




#Begin Algorithm
#Process min and max values for each column
# train_df = pd.read_csv("Datasets/NSL-KDD/KDDTrain+.txt", header=None, names=data_headers)
# max_df = train_df.max(axis=0)
# min_df = train_df.min(axis=0)
#deviation(teardrop_attack)
# fitness_df = train_df.apply(fitness, axis=1)
# print(fitness_df)
#evaluate_sample(_model, teardrop_attack)
# print("Generating Offspring")
# sample = generate_offspring(nmap_attack, nmap_attack)
# print("Original")
# print(nmap_attack)
# print(fitness(nmap_attack))
# print("Sample")
# print(sample)
# print(fitness(sample))
run_algorithm(_model, nmap_attack, 100, 20, 5)

#Ultimate goal is attack with high deviation from current attack (that is still an attack)
#Still want potential non attacks to live on around that region
#F(sample) = 0.25 * (isBenign) * deviation + 0.75 * (isAttack) * deviation
#Experiment with a b values

#Offspring Function
#Picks on scaled values based on each samples fitness
#Takes from whichever is sampled correctly
#Adds some random noise to each sample key

#Selection Function
#Top 10 fittest go into each new round