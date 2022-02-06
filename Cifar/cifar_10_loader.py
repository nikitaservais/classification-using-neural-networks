import numpy as np 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def readCifar10(small=False):    
    # si small est vrai alors ne renvoit que les 2000 premieres images
    k = 2 if small else 6
    for i in range(1,k):
        filename = 'Cifar10_' + str(i) 
        D=unpickle(filename)
        labels = D[b'labels']        
        data = D[b'data']    
        A = np.zeros( (len(labels),len(data[0])+1) )
        for j in range(len(labels)):
            A[j] = np.concatenate((np.array([ labels[j] ]), data[j]))                
        B = A if i == 1 else np.concatenate((B, A))                    
    return B[:2000] if small else B    


print(len(readCifar10(small=True)[0]))