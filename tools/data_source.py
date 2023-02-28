import numpy as np
from sklearn.model_selection import train_test_split
from .loader import get_xy


class DataSource:
    def __init__(self, proj_dir, cls_ind, train_size, wnd=(0, 0, 9999999, 9999999), mask_geojson=None, mask_class=-1):
        self.proj_dir = proj_dir
        self.wnd = wnd
        self.cls_ind = cls_ind
        self.train_size = train_size
        self.mask_geojson = mask_geojson
        self.mask_class = mask_class

    def load(self, filters_nb, layers_nb, is_inference):
        o, p, shape, cond = get_xy(self.proj_dir, self.wnd, True, filters_nb, layers_nb, is_inference, self.mask_geojson, self.mask_class)
        X, Y, shape, cond = get_xy(self.proj_dir, self.wnd, False, filters_nb, layers_nb, is_inference, self.mask_geojson, self.mask_class)
        cond_any_soft_green = (p == 1)
        cond_flat_green = (Y == 0)
        Y[np.where(cond_any_soft_green & cond_flat_green)[0]] = self.cls_ind
        I_train, I_test = train_test_split(np.arange(len(X)), random_state=0, train_size=self.train_size)
        return X[I_train], Y[I_train], X[I_test], Y[I_test]
