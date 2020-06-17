import matplotlib.pyplot as plt

#Takes in an array from our results and turns it into a latex table
def rawToLatexTable(data):
    for key in list(data.keys()):
        print(str(key) + " & " + str(data[key][0][0]) + " & " + str(data[key][0][1]) + " & " + str(data[key][0][2]))

def graphHyperparamData(data):
    x = []
    y = []
    for key in list(data.keys()):
        x.append(int(key))
        y.append(int(data[key][0][2]))
        #print(str(key) + " & " + str(data[key][0][0]) + " & " + str(data[key][0][1]) + " & " + str(data[key][0][2]))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    # title and labels, setting initial sizes
    ax.set_xlabel('Ratio of offspring produced to number of fittest offpsring to select', fontsize=12)
    ax.set_ylabel('Number of samples classified as an attack', fontsize=12)

    plt.show()