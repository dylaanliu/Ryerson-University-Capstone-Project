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

# Execute analysis of the Yelp Dataset
def yelpExecute(reviews):
    foodArray = [] # [sentiment polarity, subjectivety, date (year), food]
    foodList = ["corn bread", "rib", "ribs", "corn", "chicken", "beans",
	"steak", "beer", "root beer", "burger", "burgers", "eggs", "soup", "salad",
	"bbq", "potatoes", "toast", "penne al forno"]
    foodLabels = ['Sentiment_Score', 'Subjectivity', 'Year', 'Food']

    sentenceList = []

    # Process reviews
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

    # CountVectorizer is used to remove stop words and words that provide little
    # meaning. TfidfTransformer is used to process the sentences and give each
    # word a value (weight)
    vect = CountVectorizer()
    tfidf = TfidfTransformer()
    vX = vect.fit_transform(sentenceList)
    tfidfX = tfidf.fit_transform(vX)
    tfidfXArray = tfidfX.toarray()

    # Put the foodArray into a data frame
    foodDF = pandas.DataFrame.from_records(foodArray, columns=foodLabels)

    # Export Data Frame into Excel to create Scatter Plot
    #writer = pandas.ExcelWriter('foodDF.xlsx')
    #foodDF.to_excel(writer, 'Sheet1')
    #writer.save()

    # Correlation
    #print(foodDF.corr('pearson'))

    # Put the tfidf weights into a data frame
    tfidfXDF = pandas.DataFrame(tfidfXArray, columns=vect.get_feature_names())

    # Join foodDF and tfidfDF into one array to be used for training/testing sets
    newFoodDF = foodDF.join(tfidfXDF)

    # Convert food column in foodDF to dummies for model
    foodDF = pandas.get_dummies(newFoodDF, columns=['Food'])

    # Categorize foodDF sentiment scores
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

    # Export Data Frame into Excel to create Scatter Plot of Categorized Sentiment Scores
    #writer = pandas.ExcelWriter('foodDFCategorized.xlsx')
    #foodDF['Sentiment_Score'].to_excel(writer, 'Sheet1')
    #writer.save()

    X = foodDF.iloc[:,1:]
    y = foodDF.iloc[:,0] # labels

    # Seperate dataframe into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
                                   X, y, test_size=0.33, random_state=0)

    # Models
    # Due to lack of memory AUC graphs of the full dataset cannot be generated
    # A graph containing 1500 entries of the dataset can be found in the
    # final report
    models.yelpMultinomialRegression(X, y, X_train, X_test, y_train, y_test)
    models.yelpRFRegression(X, y, X_train, X_test, y_train, y_test)
    models.yelpSVM(X, y, X_train, X_test, y_train, y_test)

    ## OLD MODELS ##
    # Models may not be able to generate with the current training and testing
    # sets due to most recent changes to the sets
    #yelpRegression(X_train, X_test, y_train, y_test)
    #yelpDTRegression(X_train, X_test, y_train, y_test)
    #yelpBayesianRegression(X_train, X_test, y_train, y_test)
