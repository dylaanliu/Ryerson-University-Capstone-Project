import pymongo

# Establish connection to MongoDB
def dbConnect():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["yelp"]
    mycol = mydb["review"]

    return mycol
