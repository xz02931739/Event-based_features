import numpy as np

def delete_nonstage(data, label):
    """
    delete the non-stage data
    """
    data_2d = data.reshape(label.size, -1)
    data = data_2d[label != 9]
    label = label[label != 9]

    data = data.reshape(-1)

    return data, label
    