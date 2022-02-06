import numpy as np

def read_file(file):
    """
    return list of tuple with the image and the class of the image
    """
    with open (file) as f:
        data = [[int(x)/255.0 for x in line.strip().split(',')]for line in f]
        data_image = [np.reshape(x[1:], (784,1)) for x in data]
        data_y = [vecterized(x[0]*255.0) for x  in data]
        data = np.array(tuple(zip(data_image,data_y)), dtype=object)
    return data
    
def vecterized(y):
    """
    take y the class and return the vecterized form of the class
    """
    res = np.zeros((10, 1))
    res[int(y)] = 1.0
    return res