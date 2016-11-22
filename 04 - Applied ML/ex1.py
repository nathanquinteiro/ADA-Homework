import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import itertools

def run_forest_importance(data, estimators = 100, features = 0.33, depth=20):
    forest = RandomForestClassifier(n_estimators = estimators, max_features=features, max_depth=depth, n_jobs=-1)
    forest_labels = data["skin_color"]
    forest_data = data.drop(["skin_color"], axis = 1)
    train_data, test_data, train_labels, test_labels = split_data(forest_data, forest_labels, 0.7)
    forest.fit(train_data, train_labels)
    display_feature_importance(forest, forest_data, forest_labels)
    return forest, train_data, train_labels, test_data, test_labels

def display_feature_importance(forest, data, labels):
    importance = forest.feature_importances_    
    df_importance = pd.DataFrame([importance], columns = data.columns)
    n_col = len(data.columns)
    # graph
    fig, ax = plt.subplots()
    ax.bar(range(len(importance)), importance)
    ax.set_xticks(np.arange(0.5, n_col+0.5, 1))
    ax.set_xticklabels(df_importance.columns, rotation=90)
        
def split_data(data, labels, ratio=0.5):
    len_train = int(ratio*len(data))
    train_data = data.ix[:len_train,:]
    test_data = data.ix[len_train:,:]
    train_labels = labels.ix[:len_train]
    test_labels = labels.ix[len_train:]
    return train_data, test_data, train_labels, test_labels

def estimator_test(data, minTree, maxTree):
    forest_label = data["skin_color"]
    forest_data = data.drop(["skin_color"], axis = 1)

    oob = []
    for i in range(minTree, maxTree, 1):
        forest = RandomForestClassifier(n_estimators = i, max_features=0.33, max_depth=20, n_jobs=-1, oob_score=True)
        forest.fit(forest_data, forest_label)
        oob.append(forest.oob_score_)

    plt.plot(range(minTree, maxTree, 1), oob, '-')
    
    max_oob = pd.Series(oob).max()
    index_max = oob.index(max_oob)
    estimator_max = range(minTree, maxTree, 1)[index_max]
    
    return estimator_max, max_oob

def depth_test(data, minDepth, maxDepth, estimator = 100):
    forest_label = data["skin_color"]
    forest_data = data.drop(["skin_color"], axis = 1)

    oob = []
    for i in range(minDepth, maxDepth, 1):
        forest = RandomForestClassifier(n_estimators = estimator, max_features=0.33, max_depth=i, n_jobs=-1, oob_score=True)
        forest.fit(forest_data, forest_label)
        oob.append(forest.oob_score_)

    plt.plot(range(minDepth, maxDepth, 1), oob, '-')
    
    max_oob = pd.Series(oob).max()
    index_max = oob.index(max_oob)
    depth_max = range(minDepth, maxDepth, 1)[index_max]
    
    return depth_max, max_oob

def features_test(data, minFeatures, maxFeatures, estimator = 100, depth = 20):
    forest_label = data["skin_color"]
    forest_data = data.drop(["skin_color"], axis = 1)

    oob = []
    for i in range(minFeatures, maxFeatures, 1):
        forest = RandomForestClassifier(n_estimators = estimator, max_features=i, max_depth=depth, n_jobs=-1, oob_score=True)
        forest.fit(forest_data, forest_label)
        oob.append(forest.oob_score_)

    plt.plot(range(minFeatures, maxFeatures, 1), oob, '-')
    
    max_oob = pd.Series(oob).max()
    index_max = oob.index(max_oob)
    features_max = range(minFeatures, maxFeatures, 1)[index_max]
    
    return features_max, max_oob

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    
    

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 1.1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



