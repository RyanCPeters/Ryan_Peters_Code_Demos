import numpy as np

class SlidingTileWindow(object):

    def __init__(self, large_source_image:np.ndarray) -> None:
        super().__init__()
        self.source_image_reference = large_source_image
        self.bounds = self.source_image_reference.shape
        self.indices_tuple_list = []

        x_scale=self.bounds[0]
        y_scale=self.bounds[1]


