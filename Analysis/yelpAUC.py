## IMPORT ##

# Create Train/Test Sets for Model
from sklearn.model_selection import train_test_split

# Metrics
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# Plot
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from itertools import cycle

############

# AUC Multiclass reference from sklearn:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

def getAUC(X, y, clf, alg):

    y = label_binarize(y, classes=[1, 2, 3, 4, 5])
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.33, random_state=0)

    if (alg == 'rf'):
        y_score = clf.fit(X_train, y_train).predict(X_test)
    else:
        classifier = OneVsRestClassifier(clf)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    #print("worked")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for num in range(1, 6):
        print('AUC for class ', num, ' %0.2f' % roc_auc[num-1])

def getAUCPlot(X, y, clf, alg):
    y = label_binarize(y, classes=[1, 2, 3, 4, 5])
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.33, random_state=0)

    if (alg == 'rf'):
        y_score = clf.fit(X_train, y_train).predict(X_test)
    else:
        classifier = OneVsRestClassifier(clf)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    #print("worked")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print('AUC for SA < -0.6: {0:0.2f}'.format(roc_auc[0]))
    print('AUC for SA >= -0.6 and < -0.2: {0:0.2f}'.format(roc_auc[1]))
    print('AUC for SA >= -0.2 and < 0.2: {0:0.2f}'.format(roc_auc[2]))
    print('AUC for SA >= 0.2 and < 0.6: {0:0.2f}'.format(roc_auc[3]))
    print('AUC for SA >= 0.6: {0:0.2f}'.format(roc_auc[4]))
    
    # Plot

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'coral', 'crimson'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if ( alg == 'rf' ):
        plt.title('AUC for Random Forest Regression')
    elif ( alg == 'svm' ):
        plt.title('AUC for SVM')
    else:
        plt.title('AUC for Multinomial Logistic Regression')
    plt.legend(loc="lower right")
    plt.show()
