
# imports
from keras.callbacks import Callback
from matplotlib import pyplot as plt

class LivePlotBase(Callback):

    def __init__(self, dashboard_pointer:plt, plot_title:str="StubbTitle",x_range:list=range(10)):
        """

        ToDo: create a base dashboard object to act as the primary plot object that will organize and format our various plots into a single window

        :param dashboard_pointer:
        """
        super().__init__()
        self.fig = plt.figure()
        f, (self.loss_ax, self.vloss_ax ,self.acc ,self.vacc) = plt.subplots(4,1,sharex="col")
        self.loss_y = []
        self.vloss_y = []
        self.acc_y = []
        self.vacc_y = []
        self.mater_plot = plt
        self.x_range = x_range

        self.loss_ax.plot(self.x_range, self.loss_y, label="training loss")
        self.vloss_ax.plot(self.x_range,self.vloss_y,label="validation loss")
        self.acc.plot(self.x_range,self.acc_y, label="training accuracy")
        self.vacc.plot(self.x_range,self.vacc_y, label="validation accuracy")

        self.mater_plot.legend()
        self.mater_plot.tight_layout(.25)
        self.db_pointer = dashboard_pointer

    def on_batch_begin(self, batch, logs=None):
        # plt.su
        pass

