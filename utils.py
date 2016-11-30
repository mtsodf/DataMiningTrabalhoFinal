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

from sklearn.base import clone

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


def traine_with_algorithms(train_data, predictors, predict_class, alg_names, algs, log=False, balance=False):
    accuracies = []
    results = {}
    for alg, nome in zip(algs, alg_names):
        if log:
            print "Treinando %s  %s" % (nome, str(datetime.now().time()))
        accuracy = test_algorithm(
            alg, train_data, predictors, predict_class, balance=balance)
        results[nome] = accuracy
        accuracies.append(accuracy[0])

    """print "Treinando %s  %s" % ("Voting 3", str(datetime.now().time()))
    alg = VotingClassifier(estimators=[(aux[0][2], aux[0][1]), (aux[1][2], aux[1][1]), (aux[2][2], aux[2][1])])
    accuracy =  test_algorithm(alg, train_data, predictors)
    results["Voting 3"] = accuracy         """

    if log:
        for k, v in results.items():
            print "%-25s %5.3f" % (k, v)

    return results


def traine_with_selected_algorithms(train_data, predictors, predict_class, seed=1, balance=False):
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
    # algs.append(SVC(kernel='linear'))
    #algs.append(SVC(kernel='poly', coef0=2))
    #algs.append(SVC(kernel='poly', coef0=3))
    # algs.append(SVC(kernel='rbf'))
    # algs.append(SVC(kernel='sigmoid'))

    names = ["Neighbors Classifier", "Logistic Regression", "Naive-Bayes",
             "Random Forest", "Gradient Boosting", "Neural Network"]
    #names.append("SVM Linear")
    #names.append("SVM Quadratic")
    #names.append("SVM Cubic")
    #names.append("SVM RBF")
    #names.append("SVM Sigmoid")

    return traine_with_algorithms(train_data, predictors, predict_class, names, algs, balance=balance)

def sensibility(confusion_matrix):
    c = confusion_matrix
    return (float(c[1,1])/(c[1,0]+c[1,1]))

def test_algorithm(alg, train_data, predictors, predict_class, treat_output=None, seed=1, n_folds=10, balance=False):

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
        if balance:
            balanced = balance_class(train_data.iloc[train, :], predict_class)
            train_predictors = (balanced[predictors])
            train_target = balanced[predict_class].astype(float)

        else:
            train_predictors = train_data.iloc[train, :]
            train_target = train_predictors[predict_class].astype(float)
            train_predictors = train_predictors[predictors]

        # Training the algorithm using the predictors and target.
        #alg = alg.fit(train_predictors, train_target)
        alg.fit(train_predictors, train_target)
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


def balance_class(data, predict_class):

    qtd0 = len(data.loc[data[predict_class] == 0])
    qtd1 = len(data.loc[data[predict_class] == 1])

    menor_classe = 0

    if qtd1 < qtd0:
        menor_classe = 1

    adicionar = qtd1 - qtd0 if qtd1 > qtd0 else qtd0 - qtd1

    duplicate_entries = data.copy()
    menor_classe_entries = duplicate_entries.loc[
        duplicate_entries[predict_class] == menor_classe].copy()

    duplicate_entries = duplicate_entries.append(
        menor_classe_entries.sample(adicionar).copy(),  ignore_index=True)

    return duplicate_entries


def read_data(filepath):
    data = pd.read_excel(filepath)
    for i in range(2, 13):
        name = "X%d" % i
        data[name] = data[name].apply(lambda x: float(x.replace(",", ".")))
    return data


def plot_2d(train_data, predictors, predict_class, class_names, test_data=None, xlim=None, ylim=None):
    qtd_class = len(class_names)
    pca = PCA(n_components=2).fit(train_data[predictors])
    reduced_data = pca.transform(train_data[predictors])
    reduced_data = pd.DataFrame(data=reduced_data, columns=["X", "Y"])

    reduced_data[predict_class] = train_data[predict_class]

    classes = []
    for i in range(qtd_class):
        classes.append(reduced_data.loc[reduced_data[predict_class] == i])

    f, axes = plt.subplots(1, qtd_class + 1, figsize=(15, 10))

    ax = axes[0]
    ax.set_title("Todos os dados")
    colors = "rgb"
    for i in range(qtd_class):
        ax.scatter(classes[i]["X"], classes[i]["Y"], c=colors[i])

    for i in range(1, qtd_class + 1):
        axes[i].set_title(class_names[i - 1])
        axes[i].scatter(classes[i - 1]["X"], classes[i - 1]
                        ["Y"], c=colors[i - 1])

    if test_data is not None:
        reduced_data = pca.transform(test_data[predictors])
        reduced_data = pd.DataFrame(data=reduced_data, columns=["X", "Y"])
        ax.scatter(classes[i]["X"], classes[i]["Y"],
                   color='y', marker='x', linewidths=2)

    if xlim:
        for i in range(2):
            for j in range(2):
                axes[i, j].set_xlim(xlim)
    if ylim:
        for i in range(2):
            for j in range(2):
                axes[i, j].set_ylim(ylim)


def balance_class(data, predict_class):

    qtd0 = len(data.loc[data[predict_class] == 0])
    qtd1 = len(data.loc[data[predict_class] == 1])

    menor_classe = 0

    if qtd1 < qtd0:
        menor_classe = 1

    adicionar = qtd1 - qtd0 if qtd1 > qtd0 else qtd0 - qtd1

    duplicate_entries = data.copy()
    menor_classe_entries = duplicate_entries.loc[
        duplicate_entries[predict_class] == menor_classe].copy()

    duplicate_entries = duplicate_entries.append(
        menor_classe_entries.sample(adicionar).copy(),  ignore_index=True)

    return duplicate_entries


class MultiRegionClassifier:

    def __init__(self, alg, n_regions, random_state=1):

        self.n_regions = n_regions
        self.kmeans_estimator = None
        self.random_state = random_state
        self.algs = []

        for i in range(n_regions):
            self.algs.append(clone(alg))

    def fit(self, X, Y):

        df_X = pd.DataFrame(X)
        df_Y = pd.DataFrame(Y)

        self.kmeans_estimator = KMeans(
            n_clusters=self.n_regions, random_state=self.random_state)

        self.kmeans_estimator.fit(X)

        clusters = self.kmeans_estimator.predict(X)

        dict_clusters = [[] for x in range(self.n_regions)]

        df_X["clusters___"] = clusters
        df_Y["clusters___"] = clusters

        for i in range(self.n_regions):
            Xi = X.loc[X["clusters___"] == i]
            Yi = df_Y.loc[X["clusters___"] == i]

            pred_X = Xi.columns[:-1]
            pred_Y = Yi.columns[:-1]
            self.algs[i].fit(Xi[pred_X], Yi[pred_Y].astype(int))

        return self

    def predict(self, X):
        df_X = pd.DataFrame(X)

        clusters = self.kmeans_estimator.predict(df_X)

        y = []

        for i in range(len(X)):
            y.append(self.algs[clusters[i]].predict([df_X.iloc[i]])[0])

        return y


if __name__ == '__main__':
    print "Hello World!"

    data = pd.read_csv("TestDataSet.csv")
    alg = SVC(kernel='linear')
    clf = MultiRegionClassifier(alg, 2)

    clf.fit(data[["X", "Y"]], data["classe"])

    pred = pd.read_csv("PredDataSet.csv")

    print clf.predict(pred)
