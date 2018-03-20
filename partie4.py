
def read_file(file) :
    with file open as f :
        x = numpy.array([[int(s) for s in line.strip().split(',')]for line in f])
    return x
    
def sigmoide(x) :
    """
    return the sigmoide of x
    """
    try:
    	res = 1.0 / (1.0 + numpy.exp(-x))
    except OverflowError:        
        if x < 0:
            res = 0.0000000000000001
        else:
            res = 0.9999999999999999
    return res
    
class Network() :
    def __init__(self, sizes = [784, 30, 10]) :
        self.nb_layer = len(sizes)
        self.sizes = sizes
        self.weights = self.init_weights()
        
        
    def init_weights(self) :
        return numpy.array([numpy.random.normal(0,1,(x, y)) for x,y in zip(sizes[:-1],sizes[1:])])
        
    def forward_pass(self, x) :
        h = [[] for i in range(self.nb_layer)]
        a = [[] for i in range(self.nb_layer-1)]
        h[0] = x
        for i in range(1,self.nb_layer) :
            a[i] = np.dot(self.weights[i],h[i-1])
            h[i] = sigmoide(a[i])
        return (h, a)    
        
    def backpropagation(self, x, h, a) :
        """
        back propagation of a perceptron from exit neuron to input neurons
        return delta of W1 and W2
        """
        h_J = [[] for i in range(nb_layer)]
        a_J
        d_J
        g = h_J[-1] = h[self.nb_layer]-y
        for i in range(1,nb_layer,-1) :
           g = a_J[i] = g * d_sigmoide(a[i])
           d_J[i-1] = np.dot(g, h[i-1].transpose())
           g = h_J[i-1] = np.dot(self.weights[i], g)
           
        return (d_W1,d_W2)
        
        def train(self, data, p=0.5, epsilon=0.0001, H=10, learning_rate=0.001, max_it=20) :
            """
            p : probability of the dropout
            H : number of middle layers
            learning_rate : rate of which the weights are modified 
            epsilon : use for the convergence condition
            training of a perceptron on the training set data to reconize the digit.
            return the best weights found for this training
            """
            print("training...")
            numpy.random.shuffle(data)
            #init weights
            
            # init varaiables and condition
            it = 0
            max_score = 0
            condition = False
            maj = True
            # main loop
            while it < max_it and not condition and maj :
                print("it : {}".format(it))
                maj = False
                condition = False
                for x in data :
                    # forwardPass
                    z, z_out, grad, grad_out = forwardPass(x[1:], W1, W2)
                    y = check_identity(x[0], digit)
                    y_hat = 1 if z_out >= 0.5 else -1 
                    # update poids and backpropagation
                    d_W1, d_W2 = backpropagation(x_dropout, y, y_hat, z, W1_dropout, W2, grad, grad_out)
                    W1 -= learning_rate * d_W1 
                    W2 -= learning_rate * d_W2 
                    # True if at least ones the condition is reach 
                    condition = convergenceCondition(d_W1,d_W2,epsilon) or condition
                    maj = True
                score = evaluate(data, W1, W2, digit)
                if score > max_score :
                    max_score = score
                    max_W1 = numpy.copy(W1)
                    max_W2 = numpy.copy(W2)
                it += 1
            return (max_W1, max_W2)