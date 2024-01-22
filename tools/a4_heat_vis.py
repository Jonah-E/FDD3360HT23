from csv import reader
import matplotlib.pyplot as plt
import numpy as np
import sys
import pprint

def read_file(filename, index = 0):
    data = [[],[]]
    with open(filename, 'r') as csvfile:
        csvreader = reader(csvfile)
        for row in csvreader:
            data[0].append(int(row[1]))
            data[1].append(float(row[2]))
    return data

def create_plot():
    data = read_file('output_heatEq.csv', 1)
    pprint.pprint(data)

    plt.plot(data[0], data[1])
    plt.ylabel('Relative Error')
    plt.xlabel('nsteps')
    plt.show()

if __name__ == '__main__':
    create_plot()
