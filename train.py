import os
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tools.loader import get_xy


proj_dir1 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\40562'; wnd1 = (0, 0, 2100, 1800)  # all, false positive
proj_dir2 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\53508'; wnd2 = (0, 0, 8589, 4308)  # half of all, GOOD for TRAIN TREE
# proj_dir1 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\40562'; wnd1 = (0, 0, 2100//2, 1800//2)  # all, false positive
# proj_dir2 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\53508'; wnd2 = (0, 0, 8589//2, 4308//2)  # half of all, GOOD for TRAIN TREE
wnd1 = (0, 0, 5900, 10000)
wnd2 = (0, 0, 5900, 10000)


def main(solver_path, solver, filters_nb, layers_nb):
    assert layers_nb < 6

    print('Getting data...', flush=True)
    wnd1_str = '_'.join(map(str, wnd1))
    wnd2_str = '_'.join(map(str, wnd2))
    X1, Y1, shape1, cond1 = get_xy(proj_dir1, wnd1, False, filters_nb, layers_nb)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, random_state=0, train_size=0.99)
    X2, Y2, shape2, cond2 = get_xy(proj_dir2, wnd2, True, filters_nb, layers_nb)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, random_state=0, train_size=0.5)
    X_train = np.vstack([X1_train, X2_train])
    X_test = np.vstack([X1_test, X2_test])
    Y_train = np.hstack([Y1_train, Y2_train])
    Y_test = np.hstack([Y1_test, Y2_test])

    print('Data preprocessing...', flush=True)
    if True:
        scaler = preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = preprocessing.StandardScaler()  # seems this better for logistic-reg rather MinMaxScaler(even shifted)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if False:
        from sklearn.feature_selection import RFE
        rfe = RFE(solver)
        rfe = rfe.fit(X_train, Y_train)
        print(rfe.support_)
        print(rfe.ranking_)

    print('Model fitting...', flush=True)
    solver.fit(X_train, Y_train)
    tr_acc = solver.score(X_train, Y_train)
    tst_acc = solver.score(X_test, Y_test)
    tr_acc_str = f'{tr_acc:.2f}'
    tst_acc_str = f'{tst_acc:.2f}'
    print(f'Accuracy on training set: {tr_acc_str}', flush=True)
    print(f'Accuracy on test set: {tst_acc_str}', flush=True)

    solver_path = f'{solver_path}/{wnd1_str}__{wnd2_str}/{tr_acc_str}_{tst_acc_str}'
    if not os.path.exists(solver_path):
        os.makedirs(solver_path, exist_ok=True)
    print(f'Storing model to folder {solver_path}', flush=True)
    solver_fname = f'{solver_path}/solver.pkl'
    with open(solver_fname, 'wb') as f:
        pickle.dump(solver, f)
    scaler_fname = f'{solver_path}/scaler.pkl'
    with open(scaler_fname, 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    # _solver = LogisticRegression(solver='saga', random_state=0, max_iter=100)  # 0.93/0.95
    _solver = LogisticRegression()
    # _solver = DecisionTreeClassifier()
    # _solver = KNeighborsClassifier()
    # _solver = GaussianNB()
    # _solver = LinearDiscriminantAnalysis()  # 0.89/0.92(0.79)
    # _solver = QuadraticDiscriminantAnalysis(); X_train=X_train.astype(np.float32)  # 0.86/0.88(0.74)
    # _solver = SVC()  # Too Long

    # =================================================================================================================
    # filters_nb/layers_nb by log-res for mix of 2 datasets
    # 4/3->train: 0.92/0.94
    # 4/4->train: 0.92/0.94
    # 8/3(2*pi)->train: 0.86/0.88
    # 2/3->train: 0.93/0.95
    # 1/3->train: 0.94/0.96. But worse then need
    # 4/3->train: 0.92/0.94. Best compromise up to this test.(filters++)
    # 8/3->train: 0.89/0.91. Best compromise up to this test.
    # 16/3->train: 0.89/0.91. Much worse result(WHY? seed? float16?) (Filters stop)
    #  8/4->train: 0.88/0.91. (found wnd-issue) fix it for later tests
    # 8/4->train: 0.83/0.87. Unseen test terrible. Roll back... Why?
    # 8/3->train: 0.84/0.87. Even worse then previous! Fixing wnd INCREASED ONLY ONE DATASET. Has huge effect!
    # 8/4->train: 0.88/0.91. Here wnd returned to wrong values.
    #
    # filters = build_filters(8)
    # layers_nb = 4  # up to 5
    # =================================================================================================================
    _filters_nb, _layers_nb = 8, 3
    _solver_name = _solver.__class__.__name__
    _solver_path = f'{os.path.dirname(os.path.abspath(__file__))}/models/{_solver_name}/{_filters_nb}_{_layers_nb}'

    main(_solver_path, _solver, _filters_nb, _layers_nb)
    print('Finished', flush=True)
