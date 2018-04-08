import numpy as np
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def init_label(text):
    """
    init labels
    """
    label = QtGui.QLabel()
    label.setText(text)
    label.setAlignment(QtCore.Qt.AlignCenter)
    return label

def init_button(text, parent, link):
    """
    init button
    """
    button = QtGui.QPushButton(text, parent)
    button.clicked.connect(link)
    return button
    
def init_combo_box(*args):
    """
    init comboBox
    """
    comboBox = QtGui.QComboBox()
    for item in args:
        comboBox.addItem(str(item))
    return comboBox

class VisualizeWeight(QtGui.QWidget):
    def __init__(self):
        """
        Visualize the data and the weights 
        weights are shown layer by layer
        """
        super(VisualizeWeight, self).__init__()
        self._setup_layout()
        
    def _setup_layout(self):
        """
        setup layout
        """
        self.mainVLayout = QtGui.QVBoxLayout()
        self.mainGridLayout = QtGui.QGridLayout()
        self._setup_canvas()
        self._setup_mainGridLayout()
        self.mainVLayout.addLayout(self.mainGridLayout)
        self.setLayout(self.mainVLayout)
		
    def _setup_mainGridLayout(self):
        """
        setup mainGridLayout
        """
        self.right = init_button(">", self, self.right_weight)
        self.layerLabel = init_label("layer 0 to 1") 
        self.left = init_button("<", self, self.left_weight)
        self.right.setEnabled(False)
        self.left.setEnabled(False)
        self.add_box()
        
    def add_box(self):
        """
        add box to mainGridLayout
        """
        self.mainGridLayout.addWidget(self.left, 0, 3)
        self.mainGridLayout.addWidget(self.layerLabel, 0, 4)
        self.mainGridLayout.addWidget(self.right, 0, 5)
        self.mainGridLayout.addWidget(self.canvas,1,3,1,3)
        
    def _setup_canvas(self):
        """
        setup canvas
        """
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
    
    def right_weight(self):
        """
        change weights visualization
        """
        if self.it < len(self.weights)-1:
            if int(np.sqrt(self.weights[self.it+1].shape[1]))**2 == self.weights[self.it+1].shape[1] or len(self.weights[self.it+1][0]) == 3072:
                self.it += 1
                size = int(np.sqrt(self.weights[self.it].shape[1]))
                self.figure.clear()
                column = (len(self.weights[self.it])//5)
                if len(self.weights[self.it])%5 >= 1:
                    column += 1
                for i in range(len(self.weights[self.it])):
                    a = self.figure.add_subplot(5,column,i+1)
                    a.axis("off")
                    w = self.weights[self.it][i]
                    if len(self.weights[self.it][i]) == 3072:
                        w = (w-w.min())/(w.max()-w.min())
                        w = w.reshape(3,32,32).transpose([1,2,0])
                        a.imshow(w) 
                    else:
                        w = w.reshape(size, size)
                        a.imshow(w,cmap='gray') 
                self.layerLabel.setText("layers {} to {}".format(self.it,self.it+1)) 
                self.canvas.draw()
    
    def left_weight(self):
        """
        change weights visualization
        """
        if self.it > 0:    
            if int(np.sqrt(self.weights[self.it-1].shape[1]))**2 == self.weights[self.it-1].shape[1] or len(self.weights[self.it-1][0]) == 3072:
                self.it -= 1
                size = int(np.sqrt(self.weights[self.it].shape[1]))
                self.figure.clear()
                column = len(self.weights[self.it])//5
                if len(self.weights[self.it])%5 >= 1:
                    column += 1
                for i in range(len(self.weights[self.it])):
                    a = self.figure.add_subplot(5,column,i+1)
                    # a.set_title("weights "+str(i+1))
                    a.axis("off")
                    w = self.weights[self.it][i]
                    if len(self.weights[self.it][i]) == 3072:
                        w = (w-w.min())/(w.max()-w.min())
                        w = w.reshape(3,32,32).transpose([1,2,0])
                        a.imshow(w) 
                    else:
                        w = w.reshape(size,size)
                        a.imshow(w,cmap='gray')
                self.layerLabel.setText("layers {} to {}".format(self.it,self.it+1)) 
                self.canvas.draw()
    
    def get_data(self, weights):
        """
        get data from network
        """
        self.it = -1
        self.weights = weights
        self.right.setEnabled(True)
        self.left.setEnabled(True)
        self.right_weight()