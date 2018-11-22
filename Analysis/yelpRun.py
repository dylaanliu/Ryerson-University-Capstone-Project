## IMPORT ##
import pymongo
import yelp
import dbConnect
import sentence

############

db = dbConnect.dbConnect()

reviews = sentence.createReviewList("EAwh1OmG6t6p3nRaZOW_AA", db)

yelp.yelpExecute(reviews)


