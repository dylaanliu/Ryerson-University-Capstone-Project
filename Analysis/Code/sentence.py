## IMPORT ##
from nltk import tokenize, ngrams

############

# Split the reviews into sentences nltk tokenize
# and return the reviews as a list of sentences
def reviewToSentences(review):
    sentenceList = []
    sentences = tokenize.sent_tokenize(review)

    return(sentences)

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
    for x in db.find({"business_id":business_ID},
                     {"_id":0, "date":1,"stars":1,"text":1}):
        reviewList.append(x)

    return reviewList
