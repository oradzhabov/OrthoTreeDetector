import os
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import chi2_kernel
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from tools.loader import get_xy


proj_dir1 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\40562'; wnd1 = (0, 0, 2100, 1800)  # all, false positive
proj_dir2 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\53508'; wnd2 = (0, 0, 8589, 4308)  # half of all, GOOD for TRAIN TREE
#proj_dir3 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\52654';
proj_dir3 = r'D:\Program Files\Git\mnt\airzaar\execution\highwall\20861'; wnd3 = (0, 0, 59000, 10000)  # only trees
if True:
    # Big areas with proper cross-class data
    wnd1 = (0, 0, 2100, 1800)
    wnd2 = (7682, 190, 8460-7682, 3976-190)
    #wnd3 = (2166, 4941, 3804-2166, 6300-4941)
else:
    # tiny areas for quick tests. Not really proper
    wnd1_ = (990, 210, 1670-990, 880-210)
    wnd2_ = (7530, 690, 8530-7530, 2340-690)


def main(solver_path, solver, filters_nb, layers_nb, do_upsampling=False):
    assert layers_nb < 7

    print('Getting data...', flush=True)
    wnd1_str = '_'.join(map(str, wnd1))
    wnd2_str = '_'.join(map(str, wnd2))
    o1, p1, shape1, cond1 = get_xy(proj_dir1, wnd1, True, filters_nb, layers_nb)
    X1, Y1, shape1, cond1 = get_xy(proj_dir1, wnd1, False, filters_nb, layers_nb)
    Y1[np.where((p1 != 0) & (Y1 == 0))[0]] = 4
    I1_train, I1_test = train_test_split(np.arange(len(X1)), random_state=0, train_size=0.99)
    #
    X2, Y2, shape2, cond2 = get_xy(proj_dir2, wnd2, True, filters_nb, layers_nb)
    k2, l2, shape2, cond2 = get_xy(proj_dir2, wnd2, False, filters_nb, layers_nb)
    Y2[np.where((Y2 != 0) & (l2 == 0))[0]] = 2
    I2_train, I2_test = train_test_split(np.arange(len(X2)), random_state=0, train_size=0.75)

    X3, Y3, shape3, cond3 = get_xy(proj_dir3, wnd3, True, filters_nb, layers_nb)
    k3, l3, shape3, cond3 = get_xy(proj_dir3, wnd3, False, filters_nb, layers_nb)
    Y3[np.where((Y3 != 0) & (l3 == 0))[0]] = 2
    I3_train, I3_test = train_test_split(np.arange(len(X3)), random_state=0, train_size=0.75)

    # Find where new logic change markers of proj1.
    #key_pos1 = np.where((p1[I1_train] != 0) & (Y1[I1_train] == 0))[0]  # new became 0. Highlight green sloped ground
    #key_pos2 = np.where((Y2[I2_train] != 0) & (l2[I2_train] == 0))[0]  # new became 0. Highlight green tree's edges

    X_train = np.vstack([X1[I1_train], X2[I2_train], X3[I3_train]])
    X_test = np.vstack([X1[I1_test], X2[I2_test], X3[I3_test]])
    Y_train = np.hstack([Y1[I1_train], Y2[I2_train], Y3[I3_train]])
    Y_test = np.hstack([Y1[I1_test], Y2[I2_test], Y3[I3_test]])

    sample_weight = None
    if do_upsampling:
        # Resample data for over sampling minority classes
        max_samples = 0
        for i in range(5):
            if np.count_nonzero(Y_train == i) > max_samples:
                max_samples = np.count_nonzero(Y_train == i)
        X_arr = list()
        Y_arr = list()
        for i in range(5):
            class_ind = (Y_train == i)
            X_oversampled, Y_oversampled = resample(X_train[class_ind],
                                                    Y_train[class_ind],
                                                    replace=True,
                                                    n_samples=max_samples,
                                                    random_state=0)
            X_arr.append(X_oversampled)
            Y_arr.append(Y_oversampled)
        X_train = np.vstack(X_arr)
        Y_train = np.hstack(Y_arr)
    else:
        # ATTENTION: big various STD in weights make Solver slower
        sample_weight = compute_sample_weight(class_weight='balanced', y=Y_train)

    print('Data preprocessing...', flush=True)
    if False:
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

    print(f'Model fitting: DIM:{len(X_train)}*{X_train.shape[-1]}', flush=True)
    if isinstance(solver, MLPClassifier):
        solver.fit(X_train.astype(np.float32), Y_train.astype(np.float32))
    else:
        solver.fit(X_train.astype(np.float32), Y_train.astype(np.float32), sample_weight)

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
    # Solver's explanations: https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions
    # + _solver = LogisticRegression(solver='saga', random_state=0, max_iter=100)
    # _solver = LogisticRegression()
    # _solver = DecisionTreeClassifier()
    # + _solver = RandomForestClassifier(random_state=0, n_jobs=8). 3GB model, long processing. but 0.93 for new data
    _solver = MLPClassifier(random_state=0,
                            solver='adam',
                            hidden_layer_sizes=(10,),
                            max_iter=10,
                            verbose=10,
                            learning_rate_init=0.1)  # 0.91 for new data
    # _solver = KNeighborsClassifier()
    # _solver = GaussianNB()
    # _solver = LinearDiscriminantAnalysis()  # 0.89/0.92(0.79)
    # _solver = QuadraticDiscriminantAnalysis(); X_train=X_train.astype(np.float32)  # 0.86/0.88(0.74)
    # _solver = svm.SVC(kernel=chi2_kernel)  # Too Long
    # =================================================================================================================
    # filters_nb/layers_nb by log-res(saga) for mix of 2 datasets, float32, StdScaler, SampleWeights
    #      :Train/Val/Test1/Test2/Test3
    # 8/3: 0.93/0.96/0.86(bad)/0.96/0.92
    # 4/3: 0.93/0.96/0.85(bad)/0.96/0.93
    # 2/3: 0.93/0.96/0.85(bad)/0.96/0.93. Seems filters_nb has no effect
    # multi 0 from test1 by 2
    # 2/3: 0.92/0.96/0.85(tiny better, but too much false ground)/0.96/0.92
    # Add Multi all from test1 by 2
    # 2/3: 0.93/0.95/0.86(bad)/0.96/0.92
    # Add Multi all from test1 by 5
    # 2/3: 0.92/0.94/0.87/0.94/0.92
    # Introduce key-areas, and Mult them by 10
    # 2/3: 0.92/0.95/0.84(very good)/0.95(has errors, but not critical)/0.91. First SOLUTION. Probably key should be 5, not 10
    # remove not necessary weight scaler(2) and key-mult=15. It changes prod-mult from 2*10 to 15. Clever and better.
    # 2/3: 0.92/0.95/0.84(little bit smoother)/0.95/0.91(little bit worse). Need complicate the model
    #:       0.91/0.97/0.83/0.86/0.89/ for small subset
    # 2/4: 0.92/0.95/0.84/0.95/0.91. By metrics it looks like prev(2/3) checkpoint. Minor but smoothness.
    # 0/3: 0.91/0.94/0.84/0.94/0.90. Much worse but classes separated
    # 1/3: 0.91/0.95/0.84/0.95/0.90. So-so. Bit worse than 2/3.
    # Testing new features: pyramids of mean/std from cos:
    # -3/3: 0.92/0.95/0.84/0.95/0.90. SIMILAR LIKE 2/3?
    # -3/4: 0.92/0.95/0.85/0.95/0.91
    # -5/4: 0.92/0.95/0.84/0.95/0.91. Good, but has FP, as 2/3.
    # Add weights to opposite trees. Compare short datasets because FITS TOO LONG
    #:  2/3: 0.87/0.96/0.75/0.83/0.89
    # -5/4: 0.83/0.86/0.76(terrible)/0.86/0.87. Trees won. Opposite weights should be recalculated
    #:       0.87/0.95/0.79/0.83/0.89
    # Modify opposite weights(set from proj1 to proj2)
    # -5/4: 0.87/0.90/0.78(no-no-no)/0.90/0.90. Despite short acc similar to base(2/3), big became worse than big 2/3.
    #:       0.90/0.97/0.82/0.85/0.90; (like base 2/3 but much slowly fits)
    # 2/3:  0.88/0.92/0.79(still bad)/0.92/0.89. seems Gabor better than MuStd. But opposite weights make worse. Pure model?
    #:       0.90/0.97/0.80/0.86/0.90
    # Remove weights, instead add 2 extra classes for TreeEdge(2) and  GreenSlopedGround(4)
    # 2/3:  0.89/0.93/0.76(some issues)/0.89/0.84
    # -3/3: 0.90/0.93/0.76(worse then 2/3)/0.89/0.84
    # -5/3: 0.90/0.93/0.76/0.89/0.85  (quite long fitting)
    # Change training BBox to collect only proper cross-classes.
    #  2/3: 0.87/0.93/0.79/0.83/0.84(not well) (BEST)
    # Added extra training data:
    # > 2/3: 0.88/0.92/0.78(worse)/0.80/0.84(better)
    # 6/3: 0.87/0.91/0.78/0.78/0.83 (worse)
    # MLPClassifier, lr 0.1, layers 10, max iter 10
    # -3/4: 0.94/0.97/0.91(still not the best)/0.92/0.91(has issues but the best from all before)/0.84
    # =================================================================================================================
    _filters_nb, _layers_nb = -3, 4
    _solver_name = _solver.__class__.__name__
    _solver_path = f'{os.path.dirname(os.path.abspath(__file__))}/models/{_solver_name}/{_filters_nb}_{_layers_nb}'

    main(_solver_path, _solver, _filters_nb, _layers_nb, isinstance(_solver, MLPClassifier))
    print('Finished', flush=True)
# https://michael-fuchs-python.netlify.app/2021/02/03/nn-multi-layer-perceptron-classifier-mlpclassifier/