import numpy as np
from sklearn.model_selection import train_test_split
from .loader import get_xy


class DataSource:
    def __init__(self, proj_dir, sloped_is_ground_or_tree, train_size, wnd=(0, 0, 9999999, 9999999), mask_geojson=None, mask_class=-1):
        self.proj_dir = proj_dir
        self.wnd = wnd
        self.sloped_is_ground_or_tree = sloped_is_ground_or_tree
        self.train_size = train_size
        self.mask_geojson = mask_geojson
        self.mask_class = mask_class

    def load(self, filters_nb, layers_nb, is_inference, working_scale=1.0, skip_ground_class=True):
        if self.mask_geojson is None:
            X, Y, shape, cond = get_xy(self.proj_dir, self.wnd, self.sloped_is_ground_or_tree, filters_nb, layers_nb, is_inference, working_scale, skip_ground_class)
        else:
            X, Y, shape, cond = get_xy(self.proj_dir, self.wnd, False, filters_nb, layers_nb, is_inference, working_scale, skip_ground_class, self.mask_geojson, self.mask_class)
        if len(Y) == 0:
            return [], [], [], []
        I_train, I_test = train_test_split(np.arange(len(X)), random_state=0, train_size=self.train_size)
        print(f'Training samples: {len(I_train)}')
        return X[I_train], Y[I_train], X[I_test], Y[I_test]
