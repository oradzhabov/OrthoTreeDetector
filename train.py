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
from tools.data_source import DataSource

c = 1
data_src_arr = list()
# data_src_arr.append(DataSource('D:/Program Files/Git/mnt/airzaar/execution/highwall/28101', 2, 0.75*c))  # lot of light trees. Has tree.geojson
# data_src_arr.append(DataSource('D:/Program Files/Git/mnt/airzaar/execution/highwall/47269', 2, 0.75*c))  # lot of dark trees. Has tree.geojson
## data_src_arr.append(DataSource('D:/Program Files/Git/mnt/airzaar/execution/highwall/40562', 4, 0.99*c, (0, 0, 2100, 1800),'wall_32719.geojson', 4))  # green slopes
data_src_arr.append(DataSource('D:/Program Files/Git/mnt/airzaar/execution/highwall/40562', 4, 0.99*c, (0, 0, 2100, 1800)))  # green slopes
data_src_arr.append(DataSource('D:/Program Files/Git/mnt/airzaar/execution/highwall/53508', 2, 0.75*c, (7682, 190, 8460-7682, 3976-190)))  # half of all, GOOD for TRAIN TREE
data_src_arr.append(DataSource('D:/Program Files/Git/mnt/airzaar/execution/highwall/20861', 2, 0.75*c))  # only trees


def main(solver_path, solver, filters_nb, layers_nb, resampling_code):
    assert layers_nb < 7

    print('Getting data...', flush=True)
    wnd_str = '__'.join(map(str, ['_'.join(map(str, d.wnd)) for d in data_src_arr]))
    X_train = list()
    Y_train = list()
    X_test = list()
    Y_test = list()
    for d in data_src_arr:
        x_tr, y_tr, x_tst, y_tst = d.load(filters_nb, layers_nb, False)
        X_train.append(x_tr)
        Y_train.append(y_tr)
        X_test.append(x_tst)
        Y_test.append(y_tst)
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    Y_train = np.hstack(Y_train)
    Y_test = np.hstack(Y_test)

    sample_weight = None
    if resampling_code:
        # Resample data for over/under sampling minority/majority classes
        samples_count = [np.count_nonzero(Y_train == i) for i in range(5)]
        print(f'Resampling. Source samples count: {samples_count}')
        max_samples = max(samples_count)
        min_samples = min(samples_count)
        avg_samples = int(np.mean(samples_count)),
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
    if False:
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            #'hidden_layer_sizes': [(10,), (20,), (20, 10), (10, 20)],
            #'max_iter': [50, 100, 150],
            #'activation': ['tanh', 'relu'],  # tanh
            #'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            #'learning_rate_init': [0.1, 0.01, 0.001],
            #'learning_rate': ['constant', 'adaptive'],  # constant
        }
        grid = GridSearchCV(solver, param_grid, n_jobs=-1, cv=5)
        grid.fit(X_train, Y_train)
        print(grid.best_params_)
        exit(0)

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

    solver_path = f'{solver_path}/{wnd_str}/{tr_acc_str}_{tst_acc_str}'
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
                            activation='tanh',  # tuned
                            hidden_layer_sizes=(20, 10,),  # tuned
                            max_iter=200,
                            alpha=0.0001,  # tuned
                            # verbose=10,
                            learning_rate='constant',  # tuned
                            learning_rate_init=0.02)
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
    # -3/4: 0.96/0.97/0.92/0.94/0.93/0.83; Tuned params. Even better than prev.
    # > -3/4: 0.96/0.98/0.92/0.95/0.95/0.93; Fixed labeling bug
    # -3/4: 0.96/0.99/0.93/0.99/0.96/0.98; equalize hist. BAD. WRONG LABELING
    # =================================================================================================================
    _filters_nb, _layers_nb = -3, 4
    _solver_name = _solver.__class__.__name__
    _solver_path = f'{os.path.dirname(os.path.abspath(__file__))}/models/{_solver_name}/{_filters_nb}_{_layers_nb}'

    main(_solver_path, _solver, _filters_nb, _layers_nb, isinstance(_solver, MLPClassifier))
    print('Finished', flush=True)
# https://michael-fuchs-python.netlify.app/2021/02/03/nn-multi-layer-perceptron-classifier-mlpclassifier/