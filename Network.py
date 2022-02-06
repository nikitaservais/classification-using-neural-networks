import numpy as np


def sigmoid(x, prime=False):
    """
    return the sigmoid of x or sigmoid prime
    """
    if prime:
        res = sigmoid(x) * (1 - sigmoid(x))
    else:
        res = 1.0 / (1.0 + np.exp(-x))
    return res


def relu(x, prime=False):
    """
    return the relu of x or relu prime
    """
    if prime:
        res = 1 * (x > 0)
    else:
        res = x * (x > 0)
    return res


class Network:

    def __init__(self, layers=None, logistic_function=sigmoid):
        """
        The list layers contains the number of neurons in the respective
        layers of the network
        """
        self.w = None
        self.b = None
        if layers is None:
            layers = [784, 30, 10]
        self.logistic_function = logistic_function
        self.nb_layers = len(layers)
        self.layers = layers
        self.mask = [1 for _ in range(self.nb_layers)]

    def init_weights(self):
        """
        initialized bias and weights
        """
        # scaling weights vector by the square root of its number of inputs # enable to have very fast good result (
        # work well after normalize the data)
        self.b = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.w = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layers[:-1], self.layers[1:])]

    def forward_pass(self, x, p=1.0):
        """
        forward pass on an image x with the weights
        return neurons and neurons activation
        """
        # init neuron and activation
        a = [False for _ in range(self.nb_layers - 1)]
        h = [False for _ in range(self.nb_layers)]
        # activation first layer is input
        h[0] = x
        for i in range(1, self.nb_layers):
            if p != 1:
                h[i - 1] = np.multiply(self.mask[i - 1], h[i - 1])
            a[i - 1] = np.dot(self.w[i - 1], h[i - 1]) + self.b[i - 1]
            a[i - 1] /= p  # inverted dropout
            h[i] = self.logistic_function(a[i - 1])
        return a, h

    def backpropagation(self, a, h, y):
        """
        Calculate the cost for each weights and bias using cross entropy function (and not quadriatic)
        """
        # init bias cost and weights cost list
        b_J = [np.zeros(b.shape) for b in self.b]
        w_J = [np.zeros(w.shape) for w in self.w]
        # backward gradient 
        # output gradient
        g = h[-1] - y
        b_J[-1] = g
        w_J[-1] = np.dot(g, h[-2].transpose())
        for i in range(self.nb_layers - 3, -1, -1):
            g = np.dot(self.w[i + 1].transpose(), g) * self.logistic_function(a[i], prime=True)
            b_J[i] = g
            w_J[i] = np.dot(g, h[i].transpose())
        return b_J, w_J

    def sgd(self, thread, data, alpha=0.9, keep_prob=1.0, c=0, lmbda=0.0, learning_rate=0.5, epoch=20,
            monitor_training_accuracy=True, monitor_training_cost=True, monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True):
        """
        stochastic gradient descent training algorithm 
        
        data : list of tuple (image, label) alpha : training proportion over the the test set keep_prob : probability
        of keeping a neuron from the hidden layer active lmbda : regularization factor (L2 regularization) also know
        as weights decay which prevent the weights from becoming to big. c : the maximun the norm of the weights
        squared can be. works well with dropout learning_rate : rate of the algorithm is learning epoch : number of
        iteration
        
        return cost and accuracy on the training set and the evaluation set if flags are actived
        """
        size_training_set = int(len(data) * alpha)
        test_data = data[size_training_set:]
        train_data = data[:size_training_set]
        training_accuracy = []
        training_cost = []
        evaluation_accuracy = []
        evaluation_cost = []
        n = len(train_data)
        np.random.shuffle(train_data)
        # biases and weights initialized
        self.init_weights()
        # main loop
        print("training...")
        for it in range(epoch):
            # it starts
            for x, y in train_data:
                # dropout
                if keep_prob != 1.0:
                    self.mask = self.dropout(keep_prob)
                # forward pass
                a, h = self.forward_pass(x, keep_prob)
                # backward pass
                b_J, w_J = self.backpropagation(a, h, y)
                # update weights
                for i in range(len(self.w)):
                    self.b[i] = self.b[i] - learning_rate * b_J[i]
                    self.w[i] = (1 - (learning_rate * lmbda) / n) * self.w[i] - learning_rate * w_J[
                        i]  # L2 regularization
                    if c != 0:
                        var = np.sqrt(np.linalg.norm(self.w[i])) < c
            print("Epoch {} complete".format(it + 1))
            masks = [1 for _ in range(self.nb_layers)]
            if monitor_training_accuracy:
                score = self.evaluate(train_data)
                training_accuracy.append(round(score / (len(train_data)) * 100, 2))
            if monitor_training_cost:
                cost = self.total_cost(train_data, lmbda)
                training_cost.append(cost)
            if monitor_evaluation_accuracy:
                score = self.evaluate(test_data)
                evaluation_accuracy.append(round(score / (len(test_data)) * 100, 2))
                print("test set accuracy : " + str(round(score / (len(test_data)) * 100, 2)) + '%')
            if monitor_evaluation_cost:
                cost = self.total_cost(test_data, lmbda)
                evaluation_cost.append(cost)

            thread.send_data(it + 1, epoch, training_cost, evaluation_cost, training_accuracy, evaluation_accuracy)
            if thread.stop:
                thread.terminate()
        print(max(evaluation_accuracy))
        return training_accuracy, training_cost, evaluation_accuracy, evaluation_cost, self.w

    def dropout(self, keep_prob):
        """
        keep_prob : probability of keeping a neuron from the hidden layer active
        masks : listsize of number of hidden layer true and false to represent activation
        """
        masks = [1 for _ in range(self.nb_layers)]
        masks[0] = np.random.rand(self.layers[0], 1) < 0.8  # best for the input layer to have low prob of dropout
        for i in range(1, self.nb_layers - 1):
            masks[i] = np.random.rand(self.layers[i], 1) < keep_prob
        return masks

    def feed_forward(self, h):
        """
        feed an input through the network 
        return the output activation
        """
        for b, w in zip(self.b, self.w):
            h = self.logistic_function(np.dot(w, h) + b)
        return h

    def total_cost(self, data, lmbda):
        """
        lmbda : regularization factor
        return the total cost of the network on data
        """
        cost = 0.0
        for x, y in data:
            h = self.feed_forward(x)
            cost += np.sum(np.nan_to_num(-y * np.log(h) - (1 - y) * np.log(1 - h))) / len(data)
            # cost += (0.5*np.sum((y-h)**2))/len(data)
        return cost

    def evaluate(self, data):
        """
        return the score made by the network on data
        """
        score = 0
        for x, y in data:
            h = self.feed_forward(x)
            if np.argmax(h) == np.argmax(y):
                score += 1
        return score
