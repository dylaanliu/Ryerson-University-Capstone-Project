## IMPORT ##
import pymongo
import yelpExecute
import dbConnect
import sentence

############

db = dbConnect.dbConnect()

reviews = sentence.createReviewList("EAwh1OmG6t6p3nRaZOW_AA", db)

yelpExecute.yelpExecute(reviews)
