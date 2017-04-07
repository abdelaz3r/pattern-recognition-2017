import numpy as np
import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

PARAMS = {
    'hidden_layer_sizes': [10*i for i in range(1, 11)],
    'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'max_iter': [10*i for i in range(1, 11)],
    'random_state': [*range(5)]
}

GROUP_1 = ['param_' + k for k in PARAMS if k != 'random_state']
GROUP_2 = [k for k in GROUP_1 if k != 'param_max_iter']


def open_file(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(',')
            line = [int(x) for x in line]
            a, b = line[0], line[1:]
            y.append(a)
            X.append(b)
    return np.array(X), np.array(y)


def main():
    print(">>> LOADING DATA...")
    train_X, train_y = open_file('train.csv')
    # train_X, train_y = train_X[:22], train_y[:22]
    test_X, test_y = open_file('test.csv')

    clf = MLPClassifier(tol=0)
    grid = GridSearchCV(clf, PARAMS, n_jobs=4, cv=3)

    print(">>> TRAINING...")
    grid.fit(train_X, train_y)

    print(">>> BEST PARAMS:")
    print(grid.best_params_)

    best_score = grid.score(test_X, test_y)
    print(">>> SCORE ON TEST SET:")
    print(best_score)

    df = pandas.DataFrame(grid.cv_results_)
    df = df[['params', 'mean_train_score', 'mean_test_score']
            + [col for col in df.columns if col.startswith('param_')]]

    idx = df.reset_index().groupby(GROUP_1)['mean_test_score'].idxmax()
    df = df.loc[idx]
    df = df[[col for col in df.columns if col != 'random_state']]

    print(">>> DATA SAVED TO 'result.csv'")
    df.to_csv('result.csv')

    print(">>> DRAWING PLOTS")
    groups = df.groupby(GROUP_2)
    for keys, df in groups:
        x = df['param_max_iter']
        y1 = df['mean_train_score']
        y2 = df['mean_test_score']

        name = [*zip([k[len('param_'):] for k in GROUP_2], keys)]
        filename = [k + '+' + str(v) for k, v in name]
        filename = str.join('___', filename) + '.png'

        name = [k + ' = ' + str(v) for k, v in name]
        name = str.join(', ', name)

        fig = plt.figure()
        plt.plot(x, y1, label='train')
        plt.plot(x, y2, label='validation')
        plt.title(name)
        plt.axis([0, 100, 0, 1.1])
        plt.legend()
        plt.savefig(filename)
        plt.close(fig)

    print(">>> DONE!")

if __name__ == '__main__':
    main()
