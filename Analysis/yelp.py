import sentence
import pandas
from textblob import TextBlob
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# small food list for restaurant: ("EAwh1OmG6t6p3nRaZOW_AA")
def yelpExecute(reviews):
    #foodCount = 0
    #foodDict = {}  # counts number of occurences (marked as +ve)
    foodArray = [] # [time (year), food, sentiment polarity, subjectivety]
    foodList = ["corn bread", "rib", "ribs", "corn", "chicken", "beans",
	"steak", "beer", "root beer", "burger", "burgers", "eggs", "soup", "salad",
	"bbq", "potatoes", "toast", "penne al forno"]
    foodLabels = ['Sentiment Score', 'Subjectivity', 'Year', 'Food']

    for review in reviews:
        date = review["date"]
        sentenceList = sentence.reviewToSentences(review["text"])
        for sent in sentenceList:
            wordsInSentence = sentence.oneGram(sent)
            for word in wordsInSentence:
                if word in foodList:
                    newReview = []
                    newReview.append(scoreTest(sent))
                    newReview.append(subjectivityTest(sent))
                    newReview.append(int(date[:4])) # take only the year
                    newReview.append(word)
                    #foodCount += 1
                    foodArray.append(newReview)

    # Put foodArray into a data frame
    foodDF = pandas.DataFrame.from_records(foodArray, columns=foodLabels)

    # Create dummy variables for Food column
    newFoodDF = pandas.get_dummies(foodDF, columns=['Food'])
    #print(list(newFoodDF.columns))

    
    X = newFoodDF.iloc[:,1:]
    y = newFoodDF.iloc[:,0] # labels
    #X.drop(X.columns[[0]], axis=1, inplace=True)

    # Create training a test sets
    X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.33, random_state=42)

    # Algorithm Accuracy
    yelpSVM(X_train, X_test, y_train, y_test)
    yelpRegression(X_train, X_test, y_train, y_test)
    #yelpNB(X_train, X_test, y_train, y_test)
        
# Calculate the sentiment score for each sentence using
# the blobText library. The sentiment score is in the range
# of -1 to 1.
def scoreTest(sentence):
    blobSentence = TextBlob(sentence)
    return round(blobSentence.sentiment.polarity, 4)

# Calculate the subjectivity for each sentence using
# the blobText library. The subjectivity is in the range
# of 0 to 1.
def subjectivityTest(sentence):
    blobSentence = TextBlob(sentence)
    return round(blobSentence.sentiment.subjectivity, 4)

# Calculate the favorability for each food in the food
# dictionary and return a dictionary with food favorability
# values (food: favourability value)
def favorability(foodDict, foodCount):
    for food in foodDict:
        foodDict[food] = round(foodDict[food] / foodCount, 2)

    return foodDict

def yelpLogistic(X_train, X_test, y_train, y_test):

    clf = LogisticRegression(random_state=0, solver='lbfgs',
                           multi_class='multinomial').fit(X_train, y_train)
    prediction = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test) 

    #print(prediction)
    #print(y_test)
    #print(pred_proba)
    print('Logistic Regression:')
    print('Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

    print(classification_report(y_test, prediction))

def yelpSVM(X_train, X_test, y_train, y_test):

    clf = SVR().fit(X_train, y_train)
    prediction = clf.predict(X_test)
    #pred_proba = clf.predict_proba(X_test) 

    #print(prediction)
    #print(y_test)
    #print(pred_proba)
    print('SVM:')
    print('Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

    #print(classification_report(y_test, prediction))

def yelpRegression(X_train, X_test, y_train, y_test):

    clf = LinearRegression().fit(X_train, y_train)
    prediction = clf.predict(X_test)
    #pred_proba = clf.predict_proba(X_test) 

    #print(prediction)
    #print(y_test)
    #print(pred_proba)
    print('Linear Regression:')
    print('Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

    #print(classification_report(y_test, prediction))

def yelpNB(X_train, X_test, y_train, y_test):

    clf = MultinomialNB().fit(X_train, y_train)
    prediction = clf.predict(X_test)
    #pred_proba = clf.predict_proba(X_test) 

    #print(prediction)
    #print(y_test)
    #print(pred_proba)
    print('Multinomial Naive Bayes:')
    print('Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))

    #print(classification_report(y_test, prediction))

def yelpTFIDF(X_train, X_test, y_train, y_test):
    svm = Pipeline([('vectorizer', CountVectorizer(stop_words='english')),
                   ('tfidf', TfidfTransformer()),
                   ('svm', SVR())])

    svm = svm.fit(X_train, y_train)
    
# Takes a food dictionary and sorts it from greatest to least
# and returns the sorted dictionary
def rankFood(foodDict):
    ## maybe change to string format (look like a rank list)
    # print sorted dictionary from greatest to least
    return sorted(foodDict.items(), key=lambda x: -x[1])
