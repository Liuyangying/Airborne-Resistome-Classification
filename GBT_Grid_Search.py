import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from numpy import savetxt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def  readData(filename):

    "read data from resistome"

    #'resistome.type.rf.data.txt'
    data = pd.read_csv(filename, sep ='\\\t')
    data = data.drop(['SampleID'],axis=1)
    grp = pd.unique(data['Env'])
    X = data[data.columns[1:]]
    label = data[data.columns[0]]

    return grp,X, label


def main():

    grp, X, label = readData('resistome.type.rf.data.txt')

    # Hyperparameters and their potential values
    param_grid = {
        'n_estimators': [10, 15, 20, 50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 4, 5, 6],
    }

    # Create the grid search
    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5)
    grid_search.fit(X, label)

    # Best parameters and score
    print(grid_search.best_params_)
    print(grid_search.best_score_)


if __name__ == "__main__":
    main()
    print('end')