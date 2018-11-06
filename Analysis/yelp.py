import pymongo
import sentence
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression

# Initial testing function
def yelpExecute(review):
    print(review)

def findMatch(text):
    foodCount = 0
    foodDict = {}  # counts number of occurences (marked as +ve)
    foodList = ["corn bread", "rib", "ribs", "corn", "chicken", "beans",
	"steak", "beer", "root beer", "burger", "burgers", "eggs", "soup", "salad",
	"bbq", "potatoes", "toast", "penne al forno"]
    sentenceList = sentence.splitToSentences(text)

    print(sentenceList)
    for sent in sentenceList:
        #print(sent)
        wordsInSentence = sentence.oneGram(sent)
        #print(wordsInSentence)
        for word in wordsInSentence:
            if word in foodList:                
                if scoreTest(sent) > 0:
                    foodCount += 1
                    foodDict = inFoodDict(foodDict, word)
    print(foodCount)
    print(foodDict)
    return (dictToArray(favorability(foodDict, foodCount)))
    
# If the food already exists in the dictionary increment by 1 otherwise add
# it to the dictionary with value of 1
def inFoodDict(foodDict, food):
    if food in foodDict:
        foodDict[food] = foodDict[food] + 1
    else:
        foodDict[food] = 1
    return foodDict

# Convert food dictionary into an array and return an
# array containing the values from the food dictionary.
# If food does not exist in the dictionary a value of 0
# will be given 
## INCOMPLETE ##
def dictToArray(foodDict):
    foodArray = []
    for food in foodDict:
        foodArray.append(foodDict[food])

    return foodArray
        
# Calculate the sentiment score for each sentence using
# the blobText library. The sentiment score is in the range
# of -1 to 1.
def scoreTest(sentence):
    blobSentence = TextBlob(sentence)
    return blobSentence.sentiment.polarity

# Calculate the favorability for each food in the food
# dictionary and return a dictionary with food favorability
# values (food: favourability value)
def favorability(foodDict, foodCount):
    for food in foodDict:
        foodDict[food] = round(foodDict[food] / foodCount, 2)

    return foodDict

## INCOMPLETE ##
def yelpLogistic(x, y):
    a = LogisticRegression(random_state=0, multi_class='multinomial').fit(x, y)
    a.predict(x)
    
# Takes a food dictionary and sorts it from greatest to least
# and returns the sorted dictionary
def rankFood(foodDict):
    ## maybe change to string format (look like a rank list)
    # print sorted dictionary from greatest to least
    return sorted(foodDict.items(), key=lambda x: -x[1])
