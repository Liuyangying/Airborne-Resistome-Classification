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
    clf = clf_model()

    Mtrcs = []
    #set each env as positive label in turn
    for g in grp:

        y = np.zeros(label.shape)
        y[label!=g] = 0
        y[label==g] = 1
        y = np.array(y, dtype=int)

        #cross validation
        cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        Pred = []
        Pred_p = []
        Real = []
        Mtrcs_each_g = []

        for (train, test), i in zip(cv.split(X, y), range(5)):
            # probas_ = clf.fit(X.iloc[train], y[train]).predict_proba(X[test])
            clf.fit(X.iloc[train], y[train])
            y_pred = clf.predict(X.iloc[test])
            y_pred_proba = clf.predict_proba(X.iloc[test])
            mtrcs = precision_recall_fscore_support(y[test], y_pred,pos_label=1,average='macro')
            acc = accuracy_score(y[test], y_pred)
            Mtrcs_each_g.append([acc]+list(mtrcs[:-1]))
            Pred = Pred + y_pred.tolist()
            Pred_p = Pred_p + y_pred_proba.tolist()
            Real = Real + y[test].tolist()

        Mtrcs.append(Mtrcs_each_g)
        # Pred = np.asarray(Pred)
        Real = np.asarray(Real)
        Pred_p = np.asarray(Pred_p)

        #ROC plot
        metrics.RocCurveDisplay.from_predictions(
            Real,
            Pred_p[:,1],
            name=f"{g} vs the rest",
            color="darkorange",
        )

        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"One-vs-Rest ROC curves:\n{g} vs (Other groups)")
        plt.legend()
        plt.savefig(f"SVM_ROC_figure/{g}_ROC.png",dpi=600)
        plt.show()

        # # feature importance for each env
        # perm_importance = permutation_importance(svc, X_test, y_test)
        #
        # feature_names = ['feature1', 'feature2', 'feature3', ......]
        # features = np.array(feature_names)
        #
        # sorted_idx = perm_importance.importances_mean.argsort()
        # plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
        # plt.xlabel("Permutation Importance")

        # # perm_importance = permutation_importance(clf, X_test, y_test)
        # # ft = clf.coef_
        # # ft = np.transpose(ft)
        # ft_pandas = pd.DataFrame(ft, index=list(X.columns.values),columns=["feature importance"])
        # ft_pandas.to_csv(f'SVM_Feature_rank/{g}_feature_rank.csv')


    #save envaluation metrics for each env
    Mtrcs = np.asarray(Mtrcs)
    final_score = np.mean(Mtrcs,1)
    # final_score_std = np.std(Mtrcs,1)
    panda_Mtrcs = pd.DataFrame(data = final_score, index=grp.tolist(),
                            columns = ["Accuracy","Precision","Recall", "F1score"])
    panda_Mtrcs.to_csv('SVM_Feature_rank/Precision_Recall_F1.csv')


if __name__ == "__main__":
    main()
    print('end')