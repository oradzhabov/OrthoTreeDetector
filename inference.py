import os
import cv2
import pickle
import numpy as np
from tools.loader import get_xy


def main(solver_dir, proj_dir, wnd, use_old_approach, filters_nb, layers_nb):
    # X, Y, shape, cond = get_xy(proj_dir1, wnd1, False, filters_nb, layers_nb)  # 0.86(train: 0.93/0.95), 0.86(train: 0.94/0.96)[this worse than prev]; 0.84(train: 0.92/0.94). 0.83(train: 0.89/0.91)[better but still with errors], 0.80(train: 0.89/0.91)[much worse result]. 0.80(88/91)[best]. 0.73(83/87), 0.77(84/87)
    # X, Y, shape, cond = get_xy(proj_dir2, wnd2, True, filters_nb, layers_nb)  # 0.95(train: 0.93/0.95), 0.93(train: 0.94/0.96), 0.91(0.88/0.91) 0.86(83/87)
    # X, Y, shape, cond = get_xy(proj_dir3, wnd3, True, filters_nb, layers_nb)  # 0.91(train: 0.93/0.95), 0.93(train: 0.94/0.96). 0.91(train: 0.92/0.94), 0.86(0.89/0.91), 0.82(0.88/0.91), 0.66(83/87)[hmm], 0.60(84/87)[vow]. 0.82(0.88/0.91)
    X, Y, shape, cond = get_xy(proj_dir, wnd, use_old_approach, filters_nb, layers_nb, True)

    scaler_fpath = f'{solver_dir}/scaler.pkl'
    solver_fpath = f'{solver_dir}/solver.pkl'
    with open(scaler_fpath, 'rb') as f:
        scaler = pickle.load(f)
    with open(solver_fpath, 'rb') as f:
        solver = pickle.load(f)

    X = scaler.transform(X)

    # float32 force to not mapping into float64
    acc = solver.score(X.astype(np.float32), Y.astype(np.float32))
    acc_str = f'{acc:.2f}'
    print(f'Accuracy: {acc_str}', flush=True)
    Y = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    Y[cond] = solver.predict(X.astype(np.float32))
    #Y[Y == 2] = 1  # TreeEdge became tree
    #Y[Y == 4] = 0  # GreenSlopedGround became ground
    Y = Y.reshape(shape)
    cv2.imwrite('predict.png', Y/np.max(Y)*255)


if __name__ == '__main__':
    proj_dir1 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\40562'
    wnd1 = (0, 0, 2100, 1800)  # all, false positive
    proj_dir2 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\53508'
    wnd2 = (0, 0, 8589, 4308)  # half of all, GOOD for TRAIN TREE
    proj_dir3 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\52654'
    wnd3 = (0, 0, 5900, 10000)  # all pure new
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
    _checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999/0.96_0.98'  # -3/4 pyr(cos, astd) + rgb + hsv, data up-sampled, tuned MLP, fixed err  ++++
    # _checkpoint = '0_0_2100_1800__7682_190_778_3786__0_0_9999999_9999999/0.96_0.99'  # -3/4 pyr(cos, astd) + rgb + hsv, data up-sampled, tuned MLP, fixed err, update data, equalize hist  ---. Wrong Labeling
    # _checkpoint = '990_210_680_670__7530_690_1000_1650/0.91_0.95'
    _solver_dir = f'{os.path.dirname(os.path.abspath(__file__))}/models/{_solver_name}/{_filters_nb}_{_layers_nb}/{_checkpoint}'

    main(_solver_dir, proj_dir1, wnd1, False, _filters_nb, _layers_nb)
    # main(_solver_dir, proj_dir2, wnd2, True, _filters_nb, _layers_nb)
    # main(_solver_dir, proj_dir3, wnd3, True, _filters_nb, _layers_nb)  # unobserved
    # main(_solver_dir, proj_dir4, wnd4, True, _filters_nb, _layers_nb)
    # main(_solver_dir, r'D:\Program Files\Git\mnt\airzaar\execution\highwall\52521', (0, 0, 59000, 100000), True, _filters_nb, _layers_nb)  # bad data, with gaps/holes in reconstruction
    # main(_solver_dir, r'D:\Program Files\Git\mnt\airzaar\execution\highwall\53501', (0, 0, 59000, 100000), True, _filters_nb, _layers_nb)  # old filter could resolve by own
    # main(_solver_dir, r'D:\Program Files\Git\mnt\airzaar\execution\highwall\47269', (0, 0, 59000, 100000), True, _filters_nb, _layers_nb)  # lot of dark trees
    # main(_solver_dir, r'D:\Program Files\Git\mnt\airzaar\execution\highwall\28101', (0, 0, 59000, 100000), True, _filters_nb, _layers_nb)  # Lot of false positives
