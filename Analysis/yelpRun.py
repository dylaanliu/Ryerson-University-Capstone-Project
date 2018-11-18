import pymongo
import yelp
import dbConnect
import sentence


# Connect to MongoDB
db = dbConnect.dbConnect()

# another test restaurant ("jtQARsP6P-LbkyjbO1qNGg")

# Create a Review List
reviews = sentence.createReviewList("EAwh1OmG6t6p3nRaZOW_AA", db)

yelp.yelpExecute(reviews)
