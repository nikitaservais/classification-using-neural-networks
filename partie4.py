
def read_file(file) :
    with file open as f :
        x = numpy.array([[int(s) for s in line.strip().split(',')]for line in f])
    return x
    
def logistic_function() :
    
class Network() :
    def __init__(self, sizes = [784, 20, 20, 10]) :
        self.sizes = sizes
        self.weights = self.init_weights()
        
        
    def init_weights(self) :
        return numpy.array([numpy.random.normal(0,1,(x, y)) for x,y in zip(sizes[:-1],sizes[1:])])
        
    def forward_pass(self) :
        for w in self.weights :
            a = numpy.array([[scalar_product(W1[i],x)] for i in range(H)])
        