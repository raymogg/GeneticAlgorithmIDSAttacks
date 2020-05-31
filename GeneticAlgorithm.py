from labels import data_headers, attack_generation_labels
from Model import Model
import pandas as pd
import random
from tabulate import tabulate


class GeneticAlgorithm:

    def __init__(self, debug, mutation, attack, save_model):
        self.DEBUG = debug
        self.MUTATION_PERCENTAGE = mutation
        self.ATTACK = self.select_attack(attack)
        print("Seed attack for algorithm run")
        print(self.ATTACK)

        #Get our model
        print("Initialising Model")
        self.model_obj = Model(False, attack, save_model)
        self.model = self.model_obj.generate_model()

    def select_attack(self, attack_type):
        if (attack_type != 'nmap' and attack_type != 'teardrop'):
            print("Attack type not supported: ")
            print(attack_type)
            exit(0)

        test_df = pd.read_csv("Datasets/NSL-KDD/KDDTest+.txt", header=None, names=data_headers)
        #Pick a random attack of attack_type from here
        attacks = test_df.loc[test_df['attack'] == attack_type]
        sample_attack = attacks.sample(1)
        return sample_attack.values[0]


    def getModel(self):
        return self.model

    def getSeedAttack(self):
        return self.ATTACK

    #Function to test if a feature variable is valid for the algorithm execution
    #eg is this a variable the alogrithm should be able to use
    def validId(self, idx):
        if (idx == 1 or idx == 2 or idx == 3 or idx == 41 or idx == 42):
            return False
        else:
            return True

    #Function used to evaluate a single sample on the model
    def evaluate_sample(self, model, sample):
        sample_df = pd.DataFrame([sample], columns=data_headers)
        dropped = sample_df.drop(["unknown", "attack", "protocol_type", "service", "flag"], axis=1)
        #encoded_df = pd.get_dummies(dropped, columns=["protocol_type", "service", "flag"])
        pred = model.predict(dropped)
        return (pred)

    #Function to calculate the deviation of a sample from some other sample (normally the initial seed sample)
    def deviation(self, initial_sample, sample):
        total_deviation = 0.0
        for idx, val in enumerate(sample):
            #Only measure deviation on int values
            if (type(val) is int and self.validId(idx)):
                #Factor in the value range for the value
                max_value = attack_generation_labels[data_headers[idx]][-1]
                this_deviation = (abs(val - initial_sample[idx]) / max_value)
                total_deviation += this_deviation
        return total_deviation

    #Fitness Function
    def fitness(self, model, initial_sample, sample):
        attack_eval = self.evaluate_sample(model, sample)[0]
        _deviation = self.deviation(initial_sample, sample)

        if (self.DEBUG):
            print("Sample: ")
            print(sample)
            print(attack_eval)
            print(_deviation)
        
        #Original Sample has a fitness of 1
        if (_deviation == 0):
            return 1
        
        if (attack_eval == 0):
            #Extra fitness for not being an attack
            return 2 * (1.0 / _deviation)
        else:
            return (1.0 / _deviation)

    #Function to generate some mutation to produce genetic drift in the population
    def add_mutation(self, sample):
        for idx, val in enumerate(sample):
            if (not self.validId(idx)):
                #Skip protocol type, service, flag, attack and unknown feature variables
                continue
            
            #Mutate each gene or feature variable with 5% change
            rand = random.randint(0, 100)
            if (rand <= self.MUTATION_PERCENTAGE):
                #Mutate by picking from a random index within allowable range
                max_range = attack_generation_labels[data_headers[idx]][-1]
                index = random.randint(0, max_range)
                new_value = attack_generation_labels[data_headers[idx]][index]
                sample[idx] = new_value  

        return sample 

    #Function for generating an offspring given two samples
    def generate_offspring(self, sample1, sample2):
        offspring = [None] * len(sample1) #-3 as we actually r
        #37 labels, for each one take randomly from each parent
        for val in range(len(sample1)):
            if (val == 1 or val == 2 or val == 3):
                #For protocol type, service, and flag, we simply match the parents value
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
        return self.add_mutation(offspring)

    def key_extractor(self, sample):
        return sample['fitness']

    def sample_extractor(self, sample):
        return sample['sample']

    #Prints out information for  the current population
    def display_population_statistics(self, population):
        print("FITTEST SAMPLE: "  + str(population[-1]['fitness']))
        print("WEAKEST SAMPLE: "  + str(population[0]['fitness']))
        num_attacks = 0
        for sample in population:
            if (sample['attack']):
                num_attacks += 1
        print("NUMBER OF SAMPLE ATTACKS: " + str(num_attacks))
        print(" ")

    #Prints out the current population
    def display_population(self, population):
        #Output to a html table
        table = tabulate(population, tablefmt="html", headers=list(attack_generation_labels.keys()))
        table_file = open("final_samples.html","w")
        table_file.write(table)
        table_file.close()
        print("POPULATION")
        for sample in population:
            print(sample)

    #Function to run the algorithm
    #model = the model used to evaluate attack samples
    #initial_parent = the initial genetic parent used to breed offspring
    #iterations = the number of iterations / generations to breed
    #offspring_number = the number of offspring to breed per generation
    #fittest_num = the number of offspring to pick to move through to the next generation
    def run_algorithm(self, iterations, offspring_number, fittest_num):
        #Breed initial population
        print("Breeding Initial Population")
        population = []
        for i in range(offspring_number):
                population.append(self.generate_offspring(self.ATTACK, self.ATTACK))

        print("Running Genetic Algorithm")
        for j in range(iterations):
            print("GENERATION: " + str(j))
            offspring = []
            for index in range(offspring_number):
                parent1 = random.randint(0, len(population) -1)
                parent2 = random.randint(0, len(population) -1)
                offspring.append(self.generate_offspring(population[parent1], population[parent2]))
            
            #Place offspring in population
            population.extend(offspring)

            #Evaluate the fittest_num samples to go through to next population
            fittest_samples = []
            for sample in population:
                sample_fitness = self.fitness(self.model, self.ATTACK, sample)
                is_attack = self.evaluate_sample(self.model, sample)
                fittest_samples.append({'fitness': sample_fitness, 'sample': sample, 'attack': is_attack})
            fittest_samples.sort(key=self.key_extractor)
            #Trim population if too large
            if (len(fittest_samples) > fittest_num):
                population = fittest_samples[len(fittest_samples) - fittest_num: ]
            
            if self.DEBUG:
                self.display_population_statistics(population)
            raw_population = population
            population = list(map(self.sample_extractor, population))

        self.display_population(population)
        return raw_population
