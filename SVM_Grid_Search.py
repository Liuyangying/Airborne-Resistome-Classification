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
from sklearn import svm
from sklearn.model_selection import GridSearchCV



def readData(filename):

    "read data from resistome"

    #'resistome.type.rf.data.txt'
    data = pd.read_csv(filename, sep ='\\\t')
    data = data.drop(['SampleID'],axis=1)
    grp = pd.unique(data['Env'])
    X = data[data.columns[1:]]
    label = data[data.columns[0]]

    #Normalization
    # copy the data
    df_min_max_scaled = X.copy()

    # apply normalization techniques
    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
                    df_min_max_scaled[column].max() - df_min_max_scaled[column].min())


    return grp,df_min_max_scaled, label

def clf_model():
    clf = svm.SVC(kernel = "rbf",probability= True,random_state=123, class_weight='balanced')
    return clf


def main():

    grp, X, label = readData('resistome.type.rf.data.txt')
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X, label)

    print(grid_search.best_params_)
    print(grid_search.best_score_)


if __name__ == "__main__":
    main()
    print('end')