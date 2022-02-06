import pickle

import numpy as np


def unpickle(small=False):
    k = 2 if small else 6
    for i in range(1, k):
        with open('Cifar10_' + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            data_image = [np.reshape(x / 255.0, (3072, 1)) for x in dict[b'data']]
            data_y = [vecterized(x) for x in dict[b'labels']]
            data = np.array(tuple(zip(data_image, data_y)))
        res = data if i == 1 else np.concatenate((res, data))
    return res[:2000] if small else res


def vecterized(y):
    """
    take y the class and return the vecterized form of the class
    """
    res = np.zeros((10, 1))
    res[int(y)] = 1.0
    return res


# a = unpickle("Cifar10_1")[
# a = a.reshape(3,32,32).transpose([1,2,0])
# plt.imshow(a)
# plt.show()    

print(len(unpickle(small=True)))
