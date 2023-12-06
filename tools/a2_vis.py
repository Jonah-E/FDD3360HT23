from csv import reader
import matplotlib.pyplot as plt
import numpy as np
import sys

def create_bar_plot(filename: str):
    index = []
    data = {}
    x_label = ""
    with open(filename, 'r') as csvfile:
        csvreader = reader(csvfile)
        header = next(csvreader)
        x_label = header[0]

        data = {
            'Data copy (host to device)': [],
            'GPU execution': [],
            'Data copy (device to host)': []
        }

        for row in csvreader:
            index.append(row[0])
            data['Data copy (host to device)'].append(float(row[2]))
            data['GPU execution'].append(float(row[3]))
            data['Data copy (device to host)'].append(float(row[4]))

    bottom = np.zeros(len(data['GPU execution']))

    for name, times in data.items():
        print(index)
        plt.bar(index, np.array(times), label = name, bottom = bottom)
        bottom += np.array(times)
        print(times)

    plt.ylabel('Execution time (s)')
    plt.xlabel(x_label)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Give the filename of the csv file")
        sys.exit(1)

    create_bar_plot(sys.argv[1])
