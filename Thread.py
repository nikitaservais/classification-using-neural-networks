from PyQt5.QtCore import pyqtSignal, QThread


class Thread(QThread):
    # create signals to emit to widgets
    updateBar = pyqtSignal(object, object)
    updateGraphs = pyqtSignal(object, object, object, object)

    def __init__(self, stop, net, graphics_widget, visualize_data_widget, visualize_weight_widget, *args):
        """
        get the class and the parameters for the network
        """
        super(Thread, self).__init__()
        self.stop = stop
        self.net = net
        self.graphics_widget = graphics_widget
        self.visualize_data_widget = visualize_data_widget
        self.visualize_weight_widget = visualize_weight_widget
        self.arg = args

    def __del__(self):
        self.wait()

    def send_data(self, it, max_it, cost, cost_test, score, score_test):
        """
        send data to widgets
        """
        self.updateBar.emit(it, max_it)
        # if it==max_it:
        self.updateGraphs.emit(cost, cost_test, score, score_test)

    def run(self):
        """
        run the network with the parameters 
        """
        score, cost, score_test, cost_test, weights = self.net.SGD(self, self.arg[0], alpha=self.arg[1],
                                                                   keep_prob=self.arg[2], c=self.arg[3],
                                                                   lmbda=self.arg[4], learning_rate=self.arg[5],
                                                                   epoch=self.arg[6],
                                                                   monitor_training_accuracy=self.arg[7],
                                                                   monitor_training_cost=self.arg[8],
                                                                   monitor_evaluation_accuracy=self.arg[9],
                                                                   monitor_evaluation_cost=self.arg[10])
        self.visualize_data_widget.get_data(self.arg[0])
        self.visualize_weight_widget.get_data(weights)
        if self.stop:
            self.terminate()
