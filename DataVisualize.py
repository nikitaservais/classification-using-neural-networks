from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QPushButton, QComboBox, QWidget, QVBoxLayout, QGridLayout
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def init_label(text):
    """
    init labels
    """
    label = QLabel()
    label.setText(text)
    label.setAlignment(Qt.AlignCenter)
    return label

def init_button(text, parent, link):
    """
    init button
    """
    button = QPushButton(text, parent)
    button.clicked.connect(link)
    return button
    
def init_combo_box(*args):
    """
    init comboBox
    """
    comboBox = QComboBox()
    for item in args:
        comboBox.addItem(str(item))
    return comboBox

class VisualizeData(QWidget):
    def __init__(self):
        """
        Visualize the data and the weights 
        weights are shown layer by layer
        """
        super(VisualizeData, self).__init__()
        self._setup_layout()
        
    def _setup_layout(self):
        """
        setup layout
        """
        self.mainVLayout = QVBoxLayout()
        self.mainGridLayout = QGridLayout()
        self._setup_canvas()
        self._setup_mainGridLayout()
        self.mainVLayout.addLayout(self.mainGridLayout)
        self.setLayout(self.mainVLayout)
		
    def _setup_mainGridLayout(self):
        """
        setup mainGridLayout
        """
        
        self.right = init_button(">", self, self.right_data)
        self.dataLabel = init_label("Data visualization") 
        self.left = init_button("<", self, self.left_data)
        
        self.right.setEnabled(False)
        self.left.setEnabled(False)
        self.add_box()
        
    def add_box(self):
        """
        add box to mainGridLayout
        """
        self.mainGridLayout.addWidget(self.left, 0, 0)
        self.mainGridLayout.addWidget(self.dataLabel, 0, 1)
        self.mainGridLayout.addWidget(self.right, 0, 2)
        self.mainGridLayout.addWidget(self.canvas,1,0,1,3)
        
    def _setup_canvas(self):
        """
        setup canvas
        """
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.viData1 = self.figure.add_subplot(111)
        self._setup_subplot()
        
    def _setup_subplot(self):
        """
        setup subplot
        """
        self.viData1.axis("off")
    
    def right_data(self):
        """
        change image visualization
        """
        if self.it < len(self.data)-1:
            self.it += 25
            self.figure.clear()
            for i in range(25):
                a = self.figure.add_subplot(5,5,i+1)
                a.axis('off')
                x = self.data[self.it-i+1][0]
                if len(self.data[self.it-i+1][0]) > 784:
                    x = x.reshape(3,32,32).transpose([1,2,0])
                    a.imshow(x)
                else:
                    x = x.reshape(28,28)
                    a.imshow(x, cmap="gray")
            self.canvas.draw()
        
    def left_data(self):
        """
        change image visualization
        """
        if self.it > 0:
            self.it -= 25
            self.figure.clear()
            for i in range(25):
                a = self.figure.add_subplot(5,5,i+1)
                a.axis('off')
                x = self.data[self.it+i+1][0]
                if len(self.data[self.it+i+1][0]) > 784:
                    x = x.reshape(3,32,32).transpose([1,2,0])
                    a.imshow(x)
                else:
                    x = x.reshape(28,28)
                    a.imshow(x, cmap="gray")
            self.canvas.draw()
    
    def get_data(self, data):
        """
        get data from network
        """
        self.it = -25
        self.data = data
        self.right.setEnabled(True)
        self.left.setEnabled(True)
        self.right_data()