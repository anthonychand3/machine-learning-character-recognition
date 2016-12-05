from sklearn import svm
import numpy as np
import urllib


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"

# open url and store in raw_data
raw_data = urllib.urlopen(url)

# Load data from text raw_data
    # Data type float32 - Single precistion float
    # Delimiter - Values seperated by a commma(,)
    # Converters - Takes the first element in each line and converts to a integer
dataset= np.loadtxt(raw_data, dtype= 'float32', delimiter = ',', converters= {0: lambda ch: ord(ch)-ord('A')})

# Splits the dataset into two subarrays, one for training one for testing
train, test = np.vsplit(dataset,2)

# Split the training data into the Labels which is the first element in each row and the data
responses, trainData = np.hsplit(train,[1])

# Split the Testing data into the Labels which is the first element in each row and the data
labels, testData = np.hsplit(test,[1])

# remove the comma from the dataset and stores into a 1d array
trainData.ravel()


clf = svm.SVC()
print clf.fit(trainData,responses.ravel())

false = 0
correct = 0
for i in range(len(labels)):

    if(chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A'))) != chr(int(labels[i][0] + ord('A')))):
        print "This is the False prediction: " + chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A')))
        print "This is the correct answer: " + chr(int(labels[i][0] + ord('A')))
        false = false + 1

    if(chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A'))) == chr(int(labels[i][0] + ord('A')))):
        print "This is the correct prediction: " + chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A')))
        print "This is the correct answer: " + chr(int(labels[i][0] + ord('A')))
        correct = correct + 1

print "We made " + str(false) + " false predictions"

print "We made " + str(correct) + " correct predictions"
