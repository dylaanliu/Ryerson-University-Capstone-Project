## IMPORT ##
import sentence

import math

# Sentiment Analysis
from textblob import TextBlob

# Create Train/Test Sets for Model
from sklearn.model_selection import train_test_split

# Feature Creation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Dataframe
import pandas

# Regression Models
from sklearn.svm import SVR, NuSVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

############

def yelpExecute(reviews):
    foodArray = [] # [time (year), food, sentiment polarity, subjectivety]
    foodList = ["corn bread", "rib", "ribs", "corn", "chicken", "beans",
	"steak", "beer", "root beer", "burger", "burgers", "eggs", "soup", "salad",
	"bbq", "potatoes", "toast", "penne al forno"]
    foodLabels = ['Sentiment Score', 'Subjectivity', 'Year', 'Food']

    sentenceList = []
    for review in reviews:

        date = review["date"]
        sentences = sentence.reviewToSentences(review["text"])
        
        for sent in sentences:
            wordsInSentence = sentence.oneGram(sent)
            for word in wordsInSentence:
                if word in foodList:
                    newReview = []
                    newReview.append(scoreTest(sent))
                    newReview.append(subjectivityTest(sent))
                    newReview.append(int(date[:4])) # take only the year
                    newReview.append(word)
                    sentenceList.append(sent)
                    foodArray.append(newReview)


    vect = CountVectorizer()
    tfidf = TfidfTransformer()
    vX = vect.fit_transform(sentenceList)
    tfidfX = tfidf.fit_transform(vX)
    tfidfXArray = tfidfX.toarray()


    foodDF = pandas.DataFrame.from_records(foodArray, columns=foodLabels)
    tfidfXDF = pandas.DataFrame(tfidfXArray, columns=vect.get_feature_names())

    newFoodDF = foodDF.join(tfidfXDF)

    foodDF = pandas.get_dummies(newFoodDF, columns=['Food'])
    
    X = foodDF.iloc[:,1:]
    y = foodDF.iloc[:,0]
    X_train, X_test, y_train, y_test = train_test_split(
                                   X, y, test_size=0.33, random_state=42)

    yelpSVM(X_train, X_test, y_train, y_test)
    yelpRegression(X_train, X_test, y_train, y_test)
    yelpDTRegression(X_train, X_test, y_train, y_test)
    yelpBayesianRegression(X_train, X_test, y_train, y_test)
        
# Calculate the sentiment score for each sentence using
# the blobText library. The sentiment score is in the range
# of -1 to 1.
def scoreTest(sentence):
    blobSentence = TextBlob(sentence)
    return round(blobSentence.sentiment.polarity, 4)

def subjectivityTest(sentence):
    blobSentence = TextBlob(sentence)
    return round(blobSentence.sentiment.subjectivity, 4)

def yelpSVM(X_train, X_test, y_train, y_test):
    clf = NuSVR(C=10, nu=1).fit(X_train, y_train)
    prediction = clf.predict(X_test)

    print('SVM')
    print('Accuracy (R2): {:.2f}'.format(clf.score(X_test, y_test)))
    print('MAE: {:.2f}'.format(mean_absolute_error(y_test, prediction)))
    print('MSE: {:.2f}'.format(mean_squared_error(y_test, prediction)))
    print('RMSE: {:.2f}\n'.format(math.sqrt(mean_squared_error(y_test, prediction))))

def yelpRegression(X_train, X_test, y_train, y_test):
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


