#import re
from nltk import tokenize
from nltk import ngrams

# Split the reviews into sentences delimiting by (.)
# and return the reviews as a list of sentences
def splitToSentence(text):
    #print(text)
    a = text["text"]
    b = a.split(".")
    print(b)
    return(b)

# Split the reviews into sentences nltk tokenize
# and return the reviews as a list of sentences
def splitToSentences(reviews):
    sentenceList = []
	
    for review in reviews:
        #sentence = re.split("([.!?])", review)
        #sentenceList.append(sentence)
        #print(tokenize.sent_tokenize(review))
        sentences = tokenize.sent_tokenize(review)
        for x in sentences:
            sentenceList.append(x)
    return(sentenceList)

# Convert the sentence into one-grams and return a list
# of words
def oneGram(sentence):
    
    words = tokenize.word_tokenize(sentence)

    return words

# Convert the sentence into two-grams and return a list
# of two-grams
def twoGram(sentence):
    
    twoGrams = ngrams(sentence.split(), 4)

    return twoGrams

# Create a list of reviews for the business ID specified
def createReviewList(business_ID, db):
    reviewList = []
	## Remove limit in final version
    for x in db.find({"business_id":business_ID},
                     {"_id":0, "date":1,"text":1}).limit(10):
        #text = x["text"]
        reviewList.append(x)

    return reviewList

# Separate the reviews by year into lists and return all the
# reviews organized by year in a list
def sepByYear(reviews):
    reviews2011 = []
    reviews2012 = []
    reviews2013 = []
    reviews2014 = []
    reviews2015 = []
    reviews2016 = []
    for review in reviews:
        if "2011" in review["date"]:
            reviews2011.append(review["text"])
        elif "2012" in review["date"]:
            reviews2012.append(review["text"])
        elif "2013" in review["date"]:
            reviews2013.append(review["text"])
        elif "2014" in review["date"]:
            reviews2014.append(review["text"])
        elif "2015" in review["date"]:
            reviews2015.append(review["text"])
        elif "2016" in review["date"]:
            reviews2016.append(review["text"])

    reviewsByYear = [reviews2011,reviews2012,reviews2013,
                     reviews2014,reviews2015,reviews2016]

    return reviewsByYear
    
