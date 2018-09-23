from unet_callbacks.live_plotter_base import LivePlotBase


class EpochLivePlotCallback(LivePlotBase):

    def __init__(self, dashboard_pointer):
        super.__init__(self,dashboard_pointer,)
        self.


    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

