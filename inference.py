import os
import cv2
import pickle
import numpy as np
from tools.loader import get_xy


def main(solver_dir, proj_dir, wnd, use_old_approach, filters_nb, layers_nb):
    # X, Y, shape, cond = get_xy(proj_dir1, wnd1, False, filters_nb, layers_nb)  # 0.86(train: 0.93/0.95), 0.86(train: 0.94/0.96)[this worse than prev]; 0.84(train: 0.92/0.94). 0.83(train: 0.89/0.91)[better but still with errors], 0.80(train: 0.89/0.91)[much worse result]. 0.80(88/91)[best]. 0.73(83/87), 0.77(84/87)
    # X, Y, shape, cond = get_xy(proj_dir2, wnd2, True, filters_nb, layers_nb)  # 0.95(train: 0.93/0.95), 0.93(train: 0.94/0.96), 0.91(0.88/0.91) 0.86(83/87)
    # X, Y, shape, cond = get_xy(proj_dir3, wnd3, True, filters_nb, layers_nb)  # 0.91(train: 0.93/0.95), 0.93(train: 0.94/0.96). 0.91(train: 0.92/0.94), 0.86(0.89/0.91), 0.82(0.88/0.91), 0.66(83/87)[hmm], 0.60(84/87)[vow]. 0.82(0.88/0.91)
    X, Y, shape, cond = get_xy(proj_dir, wnd, use_old_approach, filters_nb, layers_nb)

    scaler_fpath = f'{solver_dir}/scaler.pkl'
    solver_fpath = f'{solver_dir}/solver.pkl'
    with open(scaler_fpath, 'rb') as f:
        scaler = pickle.load(f)
    with open(solver_fpath, 'rb') as f:
        solver = pickle.load(f)

    X = scaler.transform(X)

    acc = solver.score(X, Y)
    acc_str = f'{acc:.2f}'
    print(f'Accuracy: {acc_str}', flush=True)
    Y = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    Y[cond] = solver.predict(X)
    Y = Y.reshape(shape)
    cv2.imwrite('predict.png', Y/3*255)


if __name__ == '__main__':
    proj_dir1 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\40562'
    wnd1 = (0, 0, 2100, 1800)  # all, false positive
    proj_dir2 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\53508'
    wnd2 = (0, 0, 8589, 4308)  # half of all, GOOD for TRAIN TREE
    proj_dir3 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\52654'
    wnd3 = (0, 0, 5900, 10000)  # all pure new

    _filters_nb, _layers_nb = 8, 3
    _solver_name = 'LogisticRegression'
    _checkpoint = '0_0_2100_1800__0_0_8589_4308/0.93_0.96'
    _solver_dir = f'{os.path.dirname(os.path.abspath(__file__))}/models/{_solver_name}/{_filters_nb}_{_layers_nb}/{_checkpoint}'

    main(_solver_dir, proj_dir1, wnd1, False, _filters_nb, _layers_nb)
    # main(_solver_dir, proj_dir2, wnd2, True, _filters_nb, _layers_nb)
    # main(_solver_dir, proj_dir3, wnd3, True, _filters_nb, _layers_nb)