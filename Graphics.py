from PyQt5.QtWidgets import QProgressBar, QGridLayout, QVBoxLayout, QWidget, QPushButton, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def init_check_box(text):
    """
    init check_box
    """
    check_box = QCheckBox()
    check_box.setText(text)
    check_box.setChecked(True)
    return check_box


def init_button(text, parent, link):
    """
    init button
    """
    button = QPushButton(text, parent)
    button.clicked.connect(link)
    return button


class Graphics(QWidget):
    def __init__(self):
        """
        Plot 2 graphs that represents the cost of the network and the accuracy
        """
        super(Graphics, self).__init__()
        self.current = None
        self.epoch = None
        self._setup_layout()
        self.show()

    def _setup_layout(self):
        """
        setup layout
        """
        self.MainVLayout = QVBoxLayout()
        self.MainGridLayout = QGridLayout()
        self._setup_main_grid_layout()
        self._setup_canvas()
        self.MainVLayout.addLayout(self.MainGridLayout)
        self.MainVLayout.addWidget(self.canvas)
        self.setLayout(self.MainVLayout)

    def _setup_canvas(self):
        """
        setup canvas
        """
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.costPlot = self.figure.add_subplot(121)
        self.scorePlot = self.figure.add_subplot(122)
        self._setup_subplot()

    def _setup_subplot(self):
        """
        setup subplots
        """
        self.costPlot.grid(True)
        self.costPlot.set_xlabel('Epoch')
        self.costPlot.set_ylabel('Cost')
        self.scorePlot.grid(True)
        self.scorePlot.set_xlabel("Epoch")
        self.scorePlot.set_ylabel("Score")
        self.canvas.draw()

    def _setup_main_grid_layout(self):
        """
        setup MainGridLayout
        """
        self.clearButton = init_button("Clear graphs", self, self.clear_plot)

        self.progressBar = QProgressBar()

        self.monitorTrainingAccuracy = init_check_box("Monitor training accuracy")

        self.monitorTrainingCost = init_check_box("Monitor training cost")

        self.monitorValidationAccuracy = init_check_box("Monitor evaluation accuracy")

        self.monitorValidationCost = init_check_box("Monitor evaluation cost")
        self.add_box()

    def add_box(self):
        """
        add box to MainGridLayout
        """
        self.MainGridLayout.addWidget(self.clearButton, 0, 0)
        self.MainGridLayout.addWidget(self.progressBar, 0, 1, 1, 3)
        self.MainGridLayout.addWidget(self.monitorTrainingCost, 1, 0)
        self.MainGridLayout.addWidget(self.monitorValidationCost, 1, 1)
        self.MainGridLayout.addWidget(self.monitorTrainingAccuracy, 1, 2)
        self.MainGridLayout.addWidget(self.monitorValidationAccuracy, 1, 3)

    def draw_plot(self, cost, cost_test, score, score_test):
        """
        add data to plot 
        """
        self.clear_plot()
        if len(cost) != 0:
            self.costPlot.axis([1, self.epoch, 0, max(cost)])
            self.costPlot.plot([i for i in range(1, len(cost) + 1)], cost, "r", label="Cost training")
            self.costPlot.legend(loc="upper right")
        if len(cost_test) != 0:
            self.costPlot.axis([1, self.epoch, 0, max(cost_test)])
            self.costPlot.plot([i for i in range(1, len(cost_test) + 1)], cost_test, 'b', label="Cost evaluation")
            self.costPlot.legend(loc="upper right")
        if len(score) != 0:
            self.scorePlot.axis([1, self.epoch, min(score) - 10, 100])
            self.scorePlot.plot([i for i in range(1, len(score) + 1)], score, "r", label="Accuracy training")
            self.scorePlot.legend(loc="upper right")
        if len(score_test) != 0:
            self.scorePlot.axis([1, self.epoch, min(score_test) - 10, 100])
            self.scorePlot.plot([i for i in range(1, len(score_test) + 1)], score_test, 'b',
                                label="Accuracy evaluation")
            self.scorePlot.legend(loc="upper right")
        self.canvas.draw()

    def clear_plot(self):
        """
        clear all plot
        """
        self.scorePlot.cla()
        self.costPlot.cla()
        self._setup_subplot()
        self.canvas.draw()

    def get_data(self):
        """
        get value from the buttons
        """
        monitor_training_accuracy_bool = self.monitorTrainingAccuracy.isChecked()
        monitor_training_cost_bool = self.monitorTrainingCost.isChecked()
        monitor_validation_accuracy_bool = self.monitorValidationAccuracy.isChecked()
        monitor_validation_cost_bool = self.monitorValidationCost.isChecked()
        return (
            monitor_training_accuracy_bool, monitor_training_cost_bool, monitor_validation_accuracy_bool,
            monitor_validation_cost_bool)

    def progress(self, current, epoch):
        """
        make the progressBar progress
        """
        self.current = current
        self.epoch = epoch
        self.progressBar.setValue(int((current / epoch) * 100))
