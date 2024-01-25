from csv import reader
import matplotlib.pyplot as plt
import numpy as np
import sys
import pprint

def read_file(filename, index = 0):
    data = {}
    with open(filename, 'r') as csvfile:
        csvreader = reader(csvfile)
        for row in csvreader:
            if row[index] in data.keys():
                data[row[index]].append(float(row[2]))
            else:
                data[row[index]] = [float(row[2])]

    for key in data.keys():
        tmp = data[key]
        data[key] = {
            'mean' : np.mean(tmp),
            'std': np.std(tmp)
        }
    return data

def create_plot():
    data = {
        'con': read_file('output_heatEq_lengths_con.csv'),
        'nocon': read_file('output_heatEq_lengths_nocon.csv')
    }
    pprint.pprint(data)

    x = [int(x) for x in data['con'].keys()]
    y = [data['con'][key]['mean'] for key in data['con'].keys()]
    e = [data['con'][key]['std'] for key in data['con'].keys()]

    plt.errorbar(x, y, e, label = 'With prefetching', capsize=3)

    x = [int(x) for x in data['nocon'].keys()]
    y = [data['nocon'][key]['mean'] for key in data['nocon'].keys()]
    e = [data['nocon'][key]['std'] for key in data['nocon'].keys()]

    plt.errorbar(x, y, e, label = 'Without prefetching' , capsize=3)

    plt.ylabel(r'Time ($\mu$s)')
    plt.xlabel('Length')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    create_plot()
