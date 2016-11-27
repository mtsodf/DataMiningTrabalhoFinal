import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
# Import the linear regression class
from sklearn.linear_model import LogisticRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier

from scipy.optimize import minimize, rosen, differential_evolution


def normalize_data(train_data, columns, type="z-score"):
    train_normalized = train_data.copy()

    for p in columns:

        if type == "z-score":
            print "Normalizando %s com z-score" % p
            train_normalized[p] = scale(train_normalized[p])
        elif type == "maxmin":
            print "Normalizando %s com MaxMin" % p
            max = train_normalized[p].max()
            media = train_normalized[p].mean()
            min = train_normalized[p].min()
            train_normalized[p] = train_normalized[
                p].apply(lambda x: (x - media) / (max - min))
        else:
            print "Type of normalization not available."
    return train_normalized


def train_with_algorithms(train_data, predictors, predict_class, alg_names, algs, log=False):
    accuracies = []
    results = {}
    for alg, nome in zip(algs, alg_names):
        if log:
            print "Treinando %s  %s" % (nome, str(datetime.now().time()))
        accuracy = test_algorithm(alg, train_data, predictors, predict_class)
        results[nome] = accuracy[0]
        accuracies.append(accuracy[0])

    """print "Treinando %s  %s" % ("Voting 3", str(datetime.now().time()))
    alg = VotingClassifier(estimators=[(aux[0][2], aux[0][1]), (aux[1][2], aux[1][1]), (aux[2][2], aux[2][1])])
    accuracy =  test_algorithm(alg, train_data, predictors)
    results["Voting 3"] = accuracy         """

    if log:
        for k, v in results.items():
            print "%-25s %5.3f" % (k, v)

    return results


def traine_with_all_the_data(train_data, predictors, predict_class, seed=1):
    algs = []
    algs.append(KNeighborsClassifier(n_neighbors=5))
    algs.append(LogisticRegression(random_state=seed))
    algs.append(GaussianNB())
    algs.append(RandomForestClassifier(random_state=1,
                                       n_estimators=50, min_samples_split=4, min_samples_leaf=2))
    algs.append(GradientBoostingClassifier(
        random_state=1, n_estimators=25, max_depth=3))
    algs.append(MLPClassifier(solver='lbfgs', alpha=1e-5,
                              hidden_layer_sizes=(3, 5), random_state=1))
    #algs.append(SVC(kernel='linear'))
    #algs.append(SVC(kernel='poly', coef0=2))
    #algs.append(SVC(kernel='poly', coef0=3))
    #algs.append(SVC(kernel='rbf'))
    #algs.append(SVC(kernel='sigmoid'))

    names = ["Neighbors Classifier", "Logistic Regression", "Naive-Bayes",
             "Random Forest", "Gradient Boosting", "Neural Network"]
    #names.append("SVM Linear")
    #names.append("SVM Quadratic")
    #names.append("SVM Cubic")
    #names.append("SVM RBF")
    #names.append("SVM Sigmoid")

    return train_with_algorithms(train_data, predictors, predict_class, names, algs)


def test_algorithm(alg, train_data, predictors, predict_class, treat_output=None, seed=1, n_folds=10):

    # Initialize our algorithm class

    # Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
    # We set random_state to ensure we get the same splits every time we run
    # this.
    kf = KFold(n_splits=n_folds, random_state=seed)

    predictions = []

    acus = []

    for train, test in kf.split(train_data):
        # The predictors we're using the train the algorithm.  Note how we only
        # take the rows in the train folds.

        train_predictors = (train_data[predictors].iloc[train, :])
        # The target we're using to train the algorithm.
        train_target = train_data[predict_class].iloc[train].astype(float)

        # Training the algorithm using the predictors and target.
        alg = alg.fit(train_predictors, train_target)

        # We can now make predictions on the test fold
        test_predictions = alg.predict(train_data[predictors].iloc[test, :])

        if treat_output:
            test_predictions = treat_output(test_predictions)

        predictions.append(test_predictions)

        acus.append(accuracy_score(test_predictions, np.array(
            train_data[predict_class].iloc[test].astype(float))))

    predictions = np.concatenate(predictions, axis=0)

    confusion = confusion_matrix(np.array(
        train_data[predict_class].astype(int)), predictions, sample_weight=None)

    return sum(acus) / len(acus), acus, predictions, confusion


def read_data(filepath):
    data = pd.read_excel(filepath)
    for i in range(2, 13):
        name = "X%d" % i
        data[name] = data[name].apply(lambda x: float(x.replace(",", ".")))
    return data


def get_train_and_test(data):
    return data.loc[(data.Classe == 0) | (data.Classe == 1) | (data.Classe == 2)], data.loc[(data.Classe == 3)],


def plot_2d(train_data, predictors, predict_class, class_names, test_data=None, xlim=None, ylim=None):
    qtd_class = len(class_names)
    pca = PCA(n_components=2).fit(train_data[predictors])
    reduced_data = pca.transform(train_data[predictors])
    reduced_data = pd.DataFrame(data=reduced_data, columns=["X", "Y"])

    reduced_data[predict_class] = train_data[predict_class]
    
    classes = []
    for i in range(qtd_class):
        classes.append(reduced_data.loc[reduced_data[predict_class] == i])

    f, axes =  plt.subplots(1, qtd_class+1, figsize=(15,10))

    ax = axes[0]
    ax.set_title("Todos os dados")
    colors = "rgb"
    for i in range(qtd_class):
        ax.scatter(classes[i]["X"], classes[i]["Y"], c=colors[i])
    
    for i in range(1, qtd_class + 1):
        axes[i].set_title(class_names[i-1])
        axes[i].scatter(classes[i-1]["X"], classes[i-1]["Y"], c=colors[i-1])

    if test_data is not None:
        reduced_data = pca.transform(test_data[predictors])
        reduced_data = pd.DataFrame(data=reduced_data, columns=["X", "Y"]) 
        ax.scatter(classes[i]["X"], classes[i]["Y"], color='y', marker='x', linewidths=2)
    
    if xlim:
        for i in range(2):
            for j in range(2):
                axes[i,j].set_xlim(xlim)
    if ylim:
        for i in range(2):
            for j in range(2):        
                axes[i,j].set_ylim(ylim)

def main():
    data = read_data("classificacao-entrada.xls")
    predictors = ["X%d" % i for i in range(1, 13)]
    predictors_id = ["id"] + predictors
    seed = 1

    print data.head()

    data.loc[data["Classe"] == "C", "Classe"] = 0
    data.loc[data["Classe"] == "V", "Classe"] = 1
    data.loc[data["Classe"] == "N", "Classe"] = 2
    data.loc[data["Classe"] == "?", "Classe"] = 3

    train, test = get_train_and_test(data)

    alg = RandomForestClassifier(
        random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
    a = test_algorithm(alg, train, predictors, "Classe")
    print "Dados nao tratados %f" % a[0]

    #alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
    #a = test_algorithm(alg, train, predictors_id, "Classe")
    results = traine_with_all_the_data(train, predictors_id, "Classe")

    for method, accuracy in results.items():
        print "%-25s %5.3f" % (method, accuracy)

    print "Previsao com ID    %f" % a[0]

    data = normalize_data(data, predictors[1:], "MaxMin")

    alg = RandomForestClassifier(
        random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
    a = test_algorithm(alg, train, predictors, "Classe")
    print "Previsao normal    %f" % a[0]





def f(x):
    return x[0] * x[0] - 2 * x[0] + 4 + x[1] * x[1]


if __name__ == '__main__':
    # main()

    data = read_data("classificacao-entrada.xls")
    predictors = ["X%d" % i for i in range(1, 13)]
    predictors_id = ["id"] + predictors
    seed = 1

    data.loc[data["Classe"] == "C", "Classe"] = 0
    data.loc[data["Classe"] == "V", "Classe"] = 1
    data.loc[data["Classe"] == "N", "Classe"] = 2
    data.loc[data["Classe"] == "?", "Classe"] = 3

    train, test = get_train_and_test(data)

    predictors = ["X%d" % i for i in range(1, 13)]

    def f(x):
        print x
        return train_with_algorithms(train, predictors, "Classe", ["SVM Linear"], [SVC(kernel='linear', gamma=x[0], coef0=x[1])])["SVM Linear"]

    bounds = [(0, 1000), (-1000.0, 1000.0)]



    def f(x):
        x[0] = int(x[0])
        x[1] = int(x[1])


        predictors = ["X%d" % i for i in range(1, 13)]

        class_weight = {}

        for i in range(3):
            class_weight[i] = x[i + 2]

        alg = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=x[0], min_samples_split=x[1], min_weight_fraction_leaf=0.0,
                                     max_features=None, random_state=seed, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=class_weight, presort=False)

        return 1/train_with_algorithms(train, predictors, "Classe", ["Decision Tree"], [alg])["Decision Tree"]

    bounds = [(1, 50), (2, 200)]

    for i in range(3):
        bounds.append((0.00001, 10.0))

    def f(x):

        print x
        predictors = ["X%d" % i for i in range(1, 13)]

        class_weight = {}

        for i in range(3):
            class_weight[i] = x[i + 2]

        return 1/train_with_algorithms(train, predictors, "Classe", ["Random Forest"], [RandomForestClassifier(
            random_state=1, n_estimators=int(x[0]), max_depth=int(x[1]), class_weight=class_weight)])["Random Forest"]

    bounds = [(1, 200), (1, 50)]
    for i in range(3):
        bounds.append((0.00001, 10.0))

    print differential_evolution(f, bounds, disp=True, popsize=20)

    data = normalize_data(data, predictors[1:])
    train, test = get_train_and_test(data)

    print differential_evolution(f, bounds, disp=True, popsize=20)    
    # print minimize(f, x0, method='BFGS', tol=1e-6).x
