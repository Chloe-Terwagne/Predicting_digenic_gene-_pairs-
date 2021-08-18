"""
    Classifier : Classify by nested cross-validation
    Name : Chloe Terwagne
    Matricule : 000409683
    date : June 2021

    Python version 3.7

"""

# IMPORT ---------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import matthews_corrcoef
import time
from mrmr import mrmr_classif

# CONSTANT -------------------------------------------------------------------------------------------------------------
PATH_DATASET = "/Users/chloe/Desktop/master_thesis/master_thesis/predicting_digenic_disease_genes/data/generated_df/final_dataframes/train_sets/part_1_train_set_dataset_A_10640_16_07.csv"
REMOVE_PHENOTYPE = False  # Set to true removed phenotypic features
MRMR = False  # Set to True reduce the number of feature to NB_MRMR_FEATURES by MRMR
NB_MRMR_FEATURES = 9


# FUNCTION -------------------------------------------------------------------------------------------------------------


def gini_feature(df, importance):
    features = list(df.columns)
    indices = np.argsort(importance)

    print('\nAveraged feature importance over the 10 folds rounded to 2 decimal places -----------------------------')
    for i in range(len(indices)):
        print("| ", features[i], " = ", round(importance[i] * 100, 2))
    print("---------------------------------------------------------------------------------------------------------")


def print_metrics(dict_metric):
    print(
        "\n=============================================== Recap ====================================================")

    # Confusion matrix average
    print("mean conf matrix = \n", np.mean(dict_metric["conf_matrix"], axis=0))

    # ROC AUC
    print("Mean ROC AUC = ", np.mean(dict_metric["aucs"]))
    print("std ROC AUC = ", np.std(dict_metric["aucs"]))

    # PR AUC
    print("Mean PR AUC = ", np.mean(dict_metric["pr_aucs"]))
    print("std PR AUC = ",  np.std(dict_metric["pr_aucs"]))

    # Mattew:
    print("Mattew correlation mean = ", np.mean(dict_metric["mattew_correlation"]))
    print("Mattew correlation std = ", np.std(dict_metric["mattew_correlation"]))

    # ppv:
    print("PPV mean = ", np.mean(dict_metric["ppv_list"]))
    print("PPV std = ", np.std(dict_metric["ppv_list"]))

    # Gini plot
    gini_mean = np.mean(dict_metric["gini_importances"], axis=0)
    gini_feature(df, gini_mean)


def eval_model(X, y):
    best_mmc_score = 0
    dict_metric = {"conf_matrix": [], "gini_importances": [], "mattew_correlation": [], "ppv_list": [], "tprs": [],
                   "aucs": [], "pr_aucs": []}

    # define search space
    p_grid = {'n_estimators': [300, 400, 500, 600],
                  'max_depth': [6, 8, 10, 12]}
    n_cv_outer, n_cv_inner, test_siz = 10, 5, 0.2
    print("\nThe outer StratifiedShuffleSplit cv is", n_cv_outer, "folds")
    print("The inner StratifiedShuffleSplit cv is", n_cv_inner, "folds")
    print("In both case the test size is ", test_siz, "and random state is 42")
    print("The hyperparameters space is :", p_grid)
    # configure the cross-validation procedure
    cv_outer = StratifiedShuffleSplit(n_splits=n_cv_outer, test_size=test_siz, random_state=42)

    print("\n--------------------- Starting the evaluation -----------------------------------------------------------")
    # enumerate splits
    for i, (train_ix, test_ix) in enumerate(cv_outer.split(X, y)):
        print("------------- outer fold ", i + 1, " ---------------")
        # split data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        # configure the cross-validation procedure
        cv_inner = StratifiedShuffleSplit(n_splits=n_cv_inner, test_size=test_siz, random_state=42)

        # define the model
        rf = RandomForestClassifier(random_state=42, class_weight="balanced_subsample")

        # define search
        grid = GridSearchCV(rf, p_grid, scoring='f1', cv=cv_inner, refit=True)

        # execute search
        model = grid.fit(X_train, y_train)

        # get the best performing model fit on the whole training set
        print("The best parameters of the inner cv are", model.best_params_)
        best_model = model.best_estimator_

        # --------evaluate the inner model-------------------------
        dict_metric["gini_importances"].append(best_model.feature_importances_)
        cm = confusion_matrix(y_test, model.predict(X_test))
        dict_metric["conf_matrix"].append(cm)
        TP = cm[1][1]
        FP = cm[0][1]
        dict_metric["ppv_list"].append(TP / (TP + FP))
        dict_metric["mattew_correlation"].append(matthews_corrcoef(y_test, model.predict(X_test)))

        # ROC Curve
        dict_metric["aucs"].append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

        # PR Curve
        y_pred_proba = model.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
        dict_metric["pr_aucs"].append(auc(recall, precision))

        # keep in memory the best parameter
        if matthews_corrcoef(y_test, model.predict(X_test)) > best_mmc_score:
            b_param = model.best_params_

    print_metrics(dict_metric)
    print("Overall best parameter regarding MCC = ", b_param)
    return b_param


def build_model(X, y, best_parameters):
    print("Building the model with ", best_parameters)
    # define the model
    crf = RandomForestClassifier(random_state=42, max_depth=best_parameters['max_depth'],
                                 n_estimators=best_parameters['n_estimators'], class_weight="balanced_subsample")
    # get the model fit on the whole training set
    final_model = crf.fit(X, y)
    return final_model


def mrmr_selection(df):
    "retrun mrmr_classif is a list containing K selected features. This is a ranking, therefore, if you want to make a further selection, take the first elements of this list."

    # The first column is the classification (target) variable for each sample
    df = df[['label'] + [col for col in df.columns if col != 'label']]
    nb_features = df.shape[1] - 1
    X = df.drop(['label'], axis=1)
    y = df['label']

    # use MRMR classification
    selected_features = mrmr_classif(X, y, K=nb_features)
    print("\nFeature ranking = ", selected_features)
    return selected_features


def eval_and_build_models(df):
    """
    Return a list of best model for each df
    :param dfs:
    :param names:
    :return:
    """

    print(
        "\n--------------------- nested cross validation configuration ---------------------------------------------------------------------")
    balance_data_df = df
    X = balance_data_df.drop(['label'], axis=1)
    y = balance_data_df['label']

    # Change y and X to np array
    X = np.array(X.values.tolist())
    y = np.array(y.values.tolist())

    df_data = balance_data_df.drop(['label'], axis=1)

    # evaluation of the model by nested cross validation
    best_parameters = eval_model(X, y)

    # Building the model
    final_model = build_model(X, y, best_parameters)

    return final_model


def drop_phenotype(df):

    try:
        df = df.drop(['jaccard_s_phenotype', 'nb_phenotype_geneA', "nb_phenotype_geneB"], axis=1)
        print("Phenotypic features removed")
    except:
        print("No phenotypic features are in this dataset.")
        pass
    return df


# Change the df visualisation
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 30)

start = time.time()

print("\n\n\nDataset path : ", PATH_DATASET)

print("Loading dataset...")
df = pd.read_csv(PATH_DATASET, index_col=0)

if REMOVE_PHENOTYPE:
    drop_phenotype(df)

print("The dataset has", df.shape[0], " gene pairs and ", str(int(df.shape[1]) - 1), "features.")
df = df.fillna(df.median())
print("Nan filled with median value of the column")

if MRMR:
    print("\n MRMR feature selection -------------------------------------------------------------")
    ranking_features = mrmr_selection(df)
    list_df = []
    names_df = []

    # Rank the df by the same ranking then the columns
    df_sorted_by_feature_ranking = df[['label']]  # label always in it
    for feature_name in ranking_features:
        df_sorted_by_feature_ranking[feature_name] = df[feature_name]

    i = NB_MRMR_FEATURES
    print("The top ", len(ranking_features[:i]), " ranked features by MRMR; the features are:", ranking_features[:i])
    reduced_df = df_sorted_by_feature_ranking.iloc[:, 0:i + 2]
    eval_and_build_models(reduced_df)
else:
    eval_and_build_models(df)

seconds = time.time() - start
print('Total Time Taken:', time.strftime("%H:%M:%S", time.gmtime(seconds)))
