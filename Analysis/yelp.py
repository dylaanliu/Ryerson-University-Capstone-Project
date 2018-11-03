import pymongo
def yelpExecute(review):
    print(review)
    
def splitToSentence(text):
    print(text)
    a = text["text"]
    b = a.split(".")
    return(b)

def findMatch(text):
    myDict = {}
    myList = ["corn bread", "rib"]
    a = splitToSentence(text)
    for sentence in a:
        for food in myList:
            if sentence.find(food) == -1:
                print("no" + food)
            else:
                myDict = inFoodDict(myDict, food)
                print(myDict)
    
def inFoodDict(foodDict, food):
    if food in foodDict:
        foodDict[food] = foodDict[food] + 1
    else:
        foodDict[food] = 1
    return foodDict
