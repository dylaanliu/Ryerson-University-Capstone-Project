## IMPORT ##
import sentence
import yelpSA as sa
import yelpModels as models

import math

# Create Train/Test Sets for Model
from sklearn.model_selection import train_test_split

# Feature Creation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Dataframe
import pandas

############

def yelpExecute(reviews):
    foodArray = [] # [time (year), food, sentiment polarity, subjectivety]
    foodList = ["corn bread", "rib", "ribs", "corn", "chicken", "beans",
	"steak", "beer", "root beer", "burger", "burgers", "eggs", "soup", "salad",
	"bbq", "potatoes", "toast", "penne al forno"]
    foodLabels = ['Sentiment_Score', 'Subjectivity', 'Year', 'Food']

    sentenceList = []
    for review in reviews:

        date = review["date"]
        sentences = sentence.reviewToSentences(review["text"])
        
        for sent in sentences:
            wordsInSentence = sentence.oneGram(sent)
            for word in wordsInSentence:
                if word in foodList:
                    newReview = []
                    newReview.append(sa.scoreTest(sent))
                    newReview.append(sa.subjectivityTest(sent))
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

    for i in range(0, len(foodDF.iloc[:,0])):
        a = foodDF.iloc[i,0]
        if ( a < -0.6 ):
            foodDF.iat[i, 0] = 1
        elif ( a >= -0.6 and a < -0.2 ):
            foodDF.iat[i, 0] = 2
        elif ( a >= -0.2 and a < 0.2 ):
            foodDF.iat[i, 0] = 3
        elif ( a >= 0.2 and a < 0.6 ):
            foodDF.iat[i, 0] = 4
        else:
            foodDF.iat[i, 0] = 5

    foodDF['Sentiment_Score']=foodDF.Sentiment_Score.astype('int64').astype('category')
    #print(foodDF)
    X = foodDF.iloc[:,1:]
    y = foodDF.iloc[:,0]
    X_train, X_test, y_train, y_test = train_test_split(
                                   X, y, test_size=0.33, random_state=0)

    #models.yelpMultinomialRegression(X, y, X_train, X_test, y_train, y_test)
    models.yelpRFRegression(X, y, X_train, X_test, y_train, y_test)
    #models.yelpSVM(X, y, X_train, X_test, y_train, y_test)

    #yelpRegression(X_train, X_test, y_train, y_test)
    #yelpDTRegression(X_train, X_test, y_train, y_test)
    #yelpBayesianRegression(X_train, X_test, y_train, y_test)
