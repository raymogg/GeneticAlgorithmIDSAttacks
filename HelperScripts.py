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
#graphHyperparamData({10: [[1.5386445994852307, 0.8126708910555295, 3]], 20: [[10.261569416498993, 0.6006451372515386, 9]], 30: [[3.0, 0.5482671100745964, 12]], 40: [[3.9924994140167205, 0.58125915175922, 12]], 50: [[13.180294041784885, 0.6360912507102126, 25]], 60: [[43.0, 0.5811044745949142, 14]], 70: [[1.3215859030837005, 0.5595465991108516, 22]], 80: [[9.728887442995498, 0.5879803464374579, 23]], 90: [[7.19718309859155, 0.6180634314322984, 29]], 100: [[5.375, 0.5730420325956331, 36]], 110: [[6.725295359408255, 0.5923572582367623, 36]], 120: [[3.7413327961221325, 0.5872377043362081, 49]], 130: [[2.891979814533923, 0.5454284695797441, 42]], 140: [[13.876778783958601, 0.5954667966217037, 44]], 150: [[50.0, 0.5843126632711234, 49]], 160: [[14.285714285714285, 0.5978951666834015, 44]], 170: [[5.463786531130876, 0.5821131268184989, 74]], 180: [[3.207507949316261, 0.6266989995366897, 59]], 190: [[14.285714285714285, 0.6164053075995175, 63]]})
#graphHyperparamData({2: [[1.7917155483600098, 0.600273602372373, 8]], 3: [[4.761904761904762, 0.8012574405225742, 11]], 4: [[63.75, 0.8571428571428572, 22]], 5: [[3.2019939275267513, 1.0725601507443137, 16]], 6: [[7.96875, 0.9228127906144933, 17]], 7: [[1.6298682536936009, 1.0, 8]], 8: [[12.063061316792659, 1.1671195325465344, 18]], 9: [[7.627118644067798, 1.0829415184695599, 15]]})
#graphHyperparamData({2: [[1.7155637114347977, 0.5673776743420759, 12]], 3: [[1.47047040834175, 0.6415066492034025, 12]], 4: [[50.0, 0.9084188296596566, 20]], 5: [[50.0, 0.9384775808133472, 21]], 6: [[8.37704918032787, 0.9871845438319098, 18]], 7: [[6.538461538461538, 1.0377331705516595, 18]], 8: [[2.174468085106383, 1.0, 21]], 9: [[5.0099130700015255, 1.2184908026916523, 20]]})
rawToLatexTable({2: [[1.7155637114347977, 0.5673776743420759, 12]], 3: [[1.47047040834175, 0.6415066492034025, 12]], 4: [[50.0, 0.9084188296596566, 20]], 5: [[50.0, 0.9384775808133472, 21]], 6: [[8.37704918032787, 0.9871845438319098, 18]], 7: [[6.538461538461538, 1.0377331705516595, 18]], 8: [[2.174468085106383, 1.0, 21]], 9: [[5.0099130700015255, 1.2184908026916523, 20]]})