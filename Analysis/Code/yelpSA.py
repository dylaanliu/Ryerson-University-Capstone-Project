## IMPORT ##

# Sentiment Analysis
from textblob import TextBlob

############

# Calculate the sentiment score for each sentence using
# the blobText library. The sentiment score is in the range
# of -1 to 1.
def scoreTest(sentence):
    blobSentence = TextBlob(sentence)
    return round(blobSentence.sentiment.polarity, 4)

# Calculate the subjectivity score for each sentence using
# the blobText library. The subjectivity score is in the range
# of 0 to 1.
def subjectivityTest(sentence):
    blobSentence = TextBlob(sentence)
    return round(blobSentence.sentiment.subjectivity, 4)
