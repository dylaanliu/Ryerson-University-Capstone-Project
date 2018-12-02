## IMPORT ##

import yelpAUC as auc

# Regression Models
from sklearn.svm import SVC # SVR, NuSVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier

# Classification Report (Performance Measures)
from sklearn.metrics import classification_report

############

def yelpMultinomialRegression(X, y, X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial')
    fitCLF = clf.fit(X_train, y_train)
    prediction = fitCLF.predict(X_test)

    print('\nMultinomial Logistic Regression')
    print('Measures:')

    print(classification_report(y_test, prediction))
    #auc.getAUC(X, y, clf, 'multi')
    #auc.getAUCPlot(X, y, clf, 'multi')


def yelpRFRegression(X, y, X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=0)
    fitCLF = clf.fit(X_train, y_train)
    prediction = fitCLF.predict(X_test)

    print('\nRandom Forest Regression')
    print('Measures:')

    print(classification_report(y_test, prediction))
    #auc.getAUC(X, y, clf, 'rf')
    auc.getAUCPlot(X, y, clf, 'rf')

def yelpSVM(X, y, X_train, X_test, y_train, y_test):
    clf = SVC(gamma='auto')
    fitCLF = clf.fit(X_train, y_train)
    prediction = fitCLF.predict(X_test)
    
    print('\nSVM')
    print(classification_report(y_test, prediction))
    #auc.getAUC(X, y, clf, 'svm')
    #auc.getAUCPlot(X, y, clf, 'svm')


## OLD MODELS ##

def yelpSVMRegression(X, y, X_train, X_test, y_train, y_test):
    clf = NuSVR(C=10, nu=1).fit(X_train, y_train)
    prediction = clf.predict(X_test)
    
    print('\nSVM')
    print('Accuracy (R2): {:.2f}'.format(clf.score(X_test, y_test)))
    print('MAE: {:.2f}'.format(mean_absolute_error(y_test, prediction)))
    print('MSE: {:.2f}'.format(mean_squared_error(y_test, prediction)))
    print('RMSE: {:.2f}\n'.format(math.sqrt(mean_squared_error(y_test, prediction))))
    print('Measures:')
    print(classification_report(y_test, prediction))

def yelpLinearRegression(X_train, X_test, y_train, y_test):
    clf = LinearRegression(normalize=True).fit(X_train, y_train)
    prediction = clf.predict(X_test)

    print('Normalized Linear Regression')
    print('Accuracy (R2): {:.2f}'.format(clf.score(X_test, y_test)))
    print('MAE: {:.2f}'.format(mean_absolute_error(y_test, prediction)))
    print('MSE: {:.2f}'.format(mean_squared_error(y_test, prediction)))
    print('RMSE: {:.2f}\n'.format(math.sqrt(mean_squared_error(y_test, prediction))))

def yelpDTRegression(X_train, X_test, y_train, y_test):
    clf = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
    prediction = clf.predict(X_test)

    print('Decision Tree Regression')
    print('Accuracy (R2): {:.2f}'.format(clf.score(X_test, y_test)))
    print('MAE: {:.2f}'.format(mean_absolute_error(y_test, prediction)))
    print('MSE: {:.2f}'.format(mean_squared_error(y_test, prediction)))
    print('RMSE: {:.2f}\n'.format(math.sqrt(mean_squared_error(y_test, prediction))))

def yelpBayesianRegression(X_train, X_test, y_train, y_test):
    clf = BayesianRidge().fit(X_train, y_train)
    prediction = clf.predict(X_test)

    print('Bayesian Ridge Regression')
    print('Accuracy (R2): {:.2f}'.format(clf.score(X_test, y_test)))
    print('MAE: {:.2f}'.format(mean_absolute_error(y_test, prediction)))
    print('MSE: {:.2f}'.format(mean_squared_error(y_test, prediction)))
    print('RMSE: {:.2f}\n'.format(math.sqrt(mean_squared_error(y_test, prediction))))
    
