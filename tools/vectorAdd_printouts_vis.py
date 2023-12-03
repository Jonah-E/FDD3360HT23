from csv import reader
import matplotlib.pyplot as plt
import numpy as np

index = []
data = {}
with open('vectorAdd_printouts.csv', 'r') as csvfile:
    csvreader = reader(csvfile)
    header = next(csvreader)

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

plt.legend()
plt.show()
