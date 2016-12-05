from sklearn import svm
import numpy as np
import urllib


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"

raw_data = urllib.urlopen(url)

dataset= np.loadtxt(raw_data, dtype= 'float32', delimiter = ',', converters= {0: lambda ch: ord(ch)-ord('A')})


train, test = np.vsplit(dataset,2)
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])

trainData.ravel()


clf = svm.SVC()
print clf.fit(trainData,responses.ravel())

false = 0
correct = 0
for i in range(len(labels)):

    if(chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A'))) != chr(int(labels[i][0] + ord('A')))):
        print "this is the prediction: " + chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A')))
        print "this is the correct answer: " + chr(int(labels[i][0] + ord('A')))
        false = false + 1

    if(chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A'))) == chr(int(labels[i][0] + ord('A')))):
        print "this is the prediction: " + chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A')))
        print "this is the correct answer: " + chr(int(labels[i][0] + ord('A')))
        correct = correct + 1

print "we made " + str(false) + " false predictions"

print "we made " + str(correct) + " correct predictions"
