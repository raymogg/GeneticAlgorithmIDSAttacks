# Generating Attacks Against IDSes Using A Genetic Algorithm
The following repository lays out the code used to produce results as part of my honours thesis at UQ. For the final produced thesis, see thesis.pdf.

# Running the code
## Install
```pip3 install -r requirements.txt```

## Running from the command line
To run a basic test run of the algorithm, first alter the basicAlgorithmTest function in the AlgorithmTesting.py file. Set the parameters you wish to use for the algorithm, including the attack you wish to target. When complete, run

```python3 AlgorithmTesting.py```

This will conduct a basic run of the algorithm, produce a final attack and plot both the seed sample and the final produced attack sample.


## Creating your own scripts
In order to use the algorithm within your own python scripts, the following pseudo code is a good starting point. You may also use AlgorithmTesting.py as a reference point.

```
    def runAlgorithm():
    #Model that adheres to the ModelWrapper interface
    model = Model()
    
    #Debug=false, mutationPercentage=18
    #attack=teardrop, saveModel=true, model=model
    algorithm = GeneticAlgorithm(False, 18, "teardrop", True, model)
    
    #Run the algorithm for 20 iterations
    #120 offspring per generation
    #0 survive each generation
    finalPopulation = algorithm.run_algorithm(20, 120, 30)
    return finalPopulation
```

## Using your own model
For the purpose of this research, a decision tree classifier was used as the IDS. However you may use any model you wish as long as it is wrapped in the model interface used by the genetic algorithm. The following is the interface your model must adhere too:

```
    class ModelWrapper:
        
        """
        Predict the class of a value(s) X
        Returns an array representing the predictions 
        for each value in X, where
        1 = sample classified as an attack and
        0 = sample classified as benign
        """
        def predict(x):
            #Run model prediction
            return pass
```
By creating your own model that implements the predict method, any classifier or hybrid model can be used as an IDS for the genetic algorithm to run against.