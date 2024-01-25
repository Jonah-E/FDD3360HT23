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
        'vectorAdd': read_file('output_vectorAdd_vector_length.csv'),
        'vectorAddStream' :read_file('output_vectorAdd_stream_vector_length.csv')
    }
    pprint.pprint(data)


    vectorAdd_x = [float(key) for key in data['vectorAdd'].keys()]
    vectorAdd = [data['vectorAdd'][key]['mean'] for key in data['vectorAdd'].keys()]
    vectorAdd_e = [data['vectorAdd'][key]['std'] for key in data['vectorAdd'].keys()]

    vectorAdd_stream_x = [float(key) for key in data['vectorAddStream'].keys()]
    vectorAdd_stream = [data['vectorAddStream'][key]['mean'] for key in data['vectorAddStream'].keys()]
    vectorAdd_stream_e = [data['vectorAddStream'][key]['std'] for key in data['vectorAddStream'].keys()]

    plt.errorbar(vectorAdd_stream_x, vectorAdd_stream, vectorAdd_stream_e,  label = 'vectorAdd-stream')
    plt.errorbar(vectorAdd_x, vectorAdd, vectorAdd_e, label = 'vectorAdd')
    plt.ylabel('Execution time (s)')
    plt.xlabel('Vector Length')
    plt.legend()
    plt.show()

    data = read_file('output_vectorAdd_stream_segment_size.csv', 1)
    pprint.pprint(data)

    vectorAdd_stream_x = [float(key) for key in data.keys()]
    vectorAdd_stream = [data[key]['mean'] for key in data.keys()]
    vectorAdd_stream_e = [data[key]['std'] for key in data.keys()]

    plt.errorbar(vectorAdd_stream_x, vectorAdd_stream, vectorAdd_stream_e,  label = 'vectorAdd-stream')
    plt.ylabel('Execution time (s)')
    plt.xlabel('Segment size')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    create_plot()
