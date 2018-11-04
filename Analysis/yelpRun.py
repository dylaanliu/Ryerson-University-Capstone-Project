import pymongo
import yelp
import dbConnect

db = dbConnect.dbConnect()

total = 0
#for x in mycol.find({"business_id":"jtQARsP6P-LbkyjbO1qNGg"}, {"text":1}):
#    print(x)

review = db.find_one({"business_id":"jtQARsP6P-LbkyjbO1qNGg"}, {"_id":0,"text":1})

yelp.findMatch(review)
#use/create a word list for food
# same for sentiment words
# look through text using 1-gram and 2-gram first matching food then sentiment
# build classifier - look for an algorithm (maybe logistic regression/Naive Bayes)
# create overall sentiment value per food
# have values for each month

# try
# GCP Cloud Natural Language API
# http://www.pingshiuanchua.com/blog/post/simple-sentiment-analysis-python?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com
# to have different sentiment values assigned

# logitistic regression - do for chance of selecting certain food and then rank

# https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
# source for list of positive and negative words

# https://pypi.org/project/textblob/

# http://www.nltk.org/index.html
