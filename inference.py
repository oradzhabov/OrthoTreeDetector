import gc
import os
import cv2
import pickle
import numpy as np
from tools.loader import get_xy


def main(solver_dir, proj_dir, wnd, use_old_approach, filters_nb, layers_nb):
    X, Y, shape, cond = get_xy(proj_dir, wnd, use_old_approach, filters_nb, layers_nb, True)

    scaler_fpath = f'{solver_dir}/scaler.pkl'
    solver_fpath = f'{solver_dir}/solver.pkl'
    with open(scaler_fpath, 'rb') as f:
        scaler = pickle.load(f)
    with open(solver_fpath, 'rb') as f:
        solver = pickle.load(f)

    indices = np.arange(len(X))
    indices_subarr = np.array_split(indices, max(1, int(len(indices) / 1e+7)))
    for ind_arr in indices_subarr:
        X[ind_arr] = scaler.transform(X[ind_arr])
    gc.collect()

    # float32 force to not mapping into float64
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    Y_ = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    acc_arr = list()
    for ind_arr in indices_subarr:
        acc_arr.append(solver.score(X[ind_arr], Y[ind_arr]))
        Y_[cond[ind_arr]] = solver.predict(X[ind_arr])
    acc = sum(acc_arr) / len(acc_arr)
    acc_str = f'{acc:.2f}'
    print(f'Accuracy: {acc_str}', flush=True)
    Y_ = Y_.reshape(shape)
    cv2.imwrite('predict.png', Y_/np.max(Y_)*255)


if __name__ == '__main__':
    proj_dir1 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\40562'
    wnd1 = (0, 0, 2100999, 1800999)  # all, false positive
    proj_dir2 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\53508'
    wnd2 = (0, 0, 8589, 4308)  # half of all, GOOD for TRAIN TREE
    proj_dir3 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\52654'
    wnd3 = (0, 0, 5900999, 10000999)  # all pure new
    proj_dir4 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\20861'
    wnd4 = (0, 0, 59000, 10000)  # only trees

    _filters_nb, _layers_nb = -3, 4
    _solver_name = 'MLPClassifier'
    # _checkpoint = '0_0_2100_1800__0_0_8589_4308/0.90_0.93'
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.32_0.40'  # -5/4 only pyr(cos, astd)
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.33_0.37'  # -5/4 only pyr(cos, astd), with complex features
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.65_0.71'  # -5/4 pyr(cos, astd) + hsv
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.74_0.81'  # -5/4 pyr(cos, astd) + rgb
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.74_0.81'  # -5/4 pyr(cos, astd) + pyr(rgb)
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.81_0.84'  # -5/4 pyr(cos, astd) + rgb + hsv
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.81_0.83'  # 2/3 pyr(cos, astd) + rgb + hsv
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.32_0.41'  # -5/5 only pyr(cos, astd)
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.32_0.41'  # -3/5 only pyr(cos, astd)
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.33_0.42'  # -3/6 only pyr(cos, astd)
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/0.81_0.85'  # -3/6 pyr(cos, astd) + rgb + hsv
    # RandomForestClassifier
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786/1.00_0.97'  # 2/3 pyr(cos, astd) + rgb + hsv
    # MLPClassifier
    # _checkpoint = '0_0_2100_1800__7682_190_778_3786/0.95_0.95'  # 2/3 pyr(cos, astd) + rgb + hsv
    # _checkpoint = '0_0_2100_1800__7682_190_778_3786/0.91_0.93'  # 2/3 pyr(cos, astd) + rgb + hsv, data up-sampled, 40 layers ++
    # _checkpoint = '0_0_2100_1800__7682_190_778_3786/0.90_0.93'  # 2/3 pyr(cos, astd) + rgb + hsv, data up-sampled, 100 layers
    # _checkpoint = '0_0_2100_1800__7682_190_778_3786/0.93_0.95'  # 2/3 pyr(cos, astd) + rgb + hsv, data up-sampled, 10 layers
    # _checkpoint = '0_0_2100_1800__7682_190_778_3786/0.94_0.97'  # -3/4 pyr(cos, astd) + rgb + hsv, data up-sampled, 10 layers  +++
    # _checkpoint = '0_0_2100_1800__7682_190_778_3786/0.96_0.97'  # -3/4 pyr(cos, astd) + rgb + hsv, data up-sampled, tuned MLP  ++++
    ## _checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999/0.96_0.98'  # -3/4 pyr(cos, astd) + rgb + hsv, data up-sampled, tuned MLP, fixed err  ++++. Leader before 2023.03.02
    #
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999__0_0_9999999_9999999/0.93_0.94'  # min/0.1(-3/4 astd)
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999__0_0_9999999_9999999/0.93_0.93'  # min/0.1(-3/3 astd/astd2/astd3)
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999__0_0_9999999_9999999/0.94_0.94'  # min/0.1(-3/4 astd/astd2/astd3)
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999__0_0_9999999_9999999/0.97_0.96'  # min/0.1(-3/6 astd/astd2/astd3)
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999__0_0_9999999_9999999/0.96_0.95'  # min/0.1(-3/6///
    #_checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999__0_0_9999999_9999999/0.95_0.96'   # max/1(-3/4 astd/astd2/astd3) adam
    _checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999__0_0_9999999_9999999/0.96_0.96'  # max/1(-3/4 astd/astd2/astd3) sgd, alpha=0.001
    #
    _solver_dir = f'{os.path.dirname(os.path.abspath(__file__))}/models/{_solver_name}/{_filters_nb}_{_layers_nb}/{_checkpoint}'

    main(_solver_dir, proj_dir1, wnd1, True, _filters_nb, _layers_nb)
    # main(_solver_dir, proj_dir2, wnd2, False, _filters_nb, _layers_nb)
    # main(_solver_dir, proj_dir3, wnd3, False, _filters_nb, _layers_nb)  # unobserved
    ## main(_solver_dir, proj_dir4, wnd4, False, _filters_nb, _layers_nb)
    # main(_solver_dir, r'D:\Program Files\Git\mnt\airzaar\execution\highwall\52521', (0, 0, 59000, 100000), True, _filters_nb, _layers_nb)  # bad data, with gaps/holes in reconstruction
    # main(_solver_dir, r'D:\Program Files\Git\mnt\airzaar\execution\highwall\53501', (0, 0, 59000, 100000), True, _filters_nb, _layers_nb)  # old filter could resolve by own
    # main(_solver_dir, r'D:\Program Files\Git\mnt\airzaar\execution\highwall\47269', (0, 0, 59000, 100000), False, _filters_nb, _layers_nb)  # lot of dark trees
    # main(_solver_dir, r'D:\Program Files\Git\mnt\airzaar\execution\highwall\28101', (0, 0, 59000, 100000), False, _filters_nb, _layers_nb)  # Lot of false positives.
