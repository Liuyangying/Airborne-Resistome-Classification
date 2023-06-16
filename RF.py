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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from numpy import savetxt
from sklearn.metrics import accuracy_score


def readData(filename):

    "read data from resistome"

    #'resistome.type.rf.data.txt'
    data = pd.read_csv(filename, sep ='\\\t')
    data = data.drop(['SampleID'],axis=1)
    grp = pd.unique(data['Env'])
    X = data[data.columns[1:]]
    label = data[data.columns[0]]

    return grp,X, label

def clf_model():

    clf = RandomForestClassifier(
        n_estimators=15,
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=0,
        verbose=0,
        warm_start=False,
        class_weight='balanced'
    )

    return clf

def main():

    grp, X, label = readData('resistome.type.rf.data.txt')
    clf = clf_model()

    Mtrcs = []
    #set each env as positive label in turn

    colornames = ["red","blue","yellow","green","cyan","purple"]
    for (g,colorname) in zip(grp,colornames):

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

        # #ROC plot for each env
        # metrics.RocCurveDisplay.from_predictions(
        #     Real,
        #     Pred_p[:,1],
        #     name=f"{g} vs the rest1",
        #     color="darkorange",
        # )
        #
        # plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        # plt.axis("square")
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title(f"One-vs-Rest ROC curves:\n{g} vs (Other groups)")
        # plt.legend()
        # # plt.savefig(f"RF_ROC_figure/12/{g}_ROC_12.png",dpi=600)
        # plt.show()

        #ROC plot for all envs
        fpr, tpr,thresholds = metrics.roc_curve(Real,Pred_p[:,1], pos_label =1)

        plt.plot(fpr, tpr, lw =2, label = '{}(AUC={:.3f})'.format(g,metrics.auc(fpr,tpr)),
                 color=colorname )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis('square')
    plt.xlim([-0.01,1.02])
    plt.ylim([-0.01,1.02])
    plt.xlabel("False Positive Rate",fontsize=14)
    plt.ylabel("True Positive Rate",fontsize=14)
    plt.title("ROC Curve",fontsize=14)
    plt.legend(loc='lower right',fontsize=9)

    plt.savefig("ROC_curve.pdf", dpi=600)
    plt.show()





        # # feature importance for each env
        # ft = clf.feature_importances_
        # ft_pandas = pd.DataFrame(ft, index=list(X.columns.values),columns=["feature importance"])
        # ft_pandas.to_csv(f'RF_Feature importance/12/{g}_feature_rank_12.csv')

    #save envaluation metrics for each env
    # Mtrcs = np.asarray(Mtrcs)
    # final_score = np.mean(Mtrcs,1)
    # # final_score_std = np.std(Mtrcs,1)
    # panda_Mtrcs = pd.DataFrame(data = final_score, index=grp.tolist(),
    #                         columns = ["Accuracy","Precision","Recall", "F1score"])
    # panda_Mtrcs.to_csv('RF_Feature importance/12/Precision_Recall_F1_12.csv')


if __name__ == "__main__":
    main()
    print('end')