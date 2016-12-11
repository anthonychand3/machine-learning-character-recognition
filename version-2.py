'''
Anthony chand
James Harrison
machine-learning-letter-recongnition
'''
from sklearn import svm
import numpy as np
import urllib

# url to dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"

# open url and store in raw_data
raw_data = urllib.urlopen(url)

# Load data from text raw_data
    # Data type float32 - Single precistion float
    # Delimiter - Values seperated by a commma(,)
    # Converters - Takes the first element in each line and converts to a integer
dataset= np.loadtxt(raw_data, dtype= 'float32', delimiter = ',', converters= {0: lambda ch: ord(ch)-ord('A')})

# Splits the dataset into two subarrays, one for training one for testing
SetOne, SetTwo, SetThree, SetFour = np.vsplit(dataset,4)

# Split the training data into the Labels which is the first element in each row and the data
LabelSetOne, DataSetOne = np.hsplit(SetOne,[1])

# Split the Testing data into the Labels which is the first element in each row and the data
LabelSetTwo, DataSetTwo = np.hsplit(SetTwo,[1])

# Split the training data into the Labels which is the first element in each row and the data
LabelSetThree, DataSetThree = np.hsplit(SetThree,[1])

# Split the Testing data into the Labels which is the first element in each row and the data
LabelSetFour, DataSetFour = np.hsplit(SetFour,[1])

training = [DataSetOne]
testing = [DataSetOne,DataSetTwo,DataSetThree,DataSetFour]
LabelSets = [LabelSetOne,LabelSetTwo,LabelSetThree,LabelSetFour]
GammaValue = [.01, 0.1, 0.5]

'''
for partition one we will test against the other
four partitions which includes its self.
'''
for x, element in enumerate(training):
    for y, elements in enumerate(testing):
        for testval in GammaValue:

            # setting vector support machine up and train using linear kernel
            clf = svm.SVC(kernel = 'linear',gamma = testval)
            print clf.fit(element,LabelSets[x].ravel())
            # confusion matrix
            Matrix = [[0 for i in range(26)] for j in range(26)]
            # prediction counters
            false = 0
            correct = 0
            
            # testing all the elements in a partition to test the prediction model
            for i in range(len(LabelSets[y])):
                if(int(clf.predict(np.array((testing[y])[i], ndmin=2))[0]) != int((LabelSets[y])[i][0])):
                    Matrix[int(clf.predict(np.array((testing[y])[i], ndmin=2))[0])][int((LabelSets[y])[i][0])] += 1
                    false = false + 1
                if(int(clf.predict(np.array((testing[y])[i], ndmin=2))[0]) == int((LabelSets[y])[i][0])):
                    Matrix[int(clf.predict(np.array((testing[y])[i], ndmin=2))[0])][int((LabelSets[y])[i][0])] += 1
                    correct = correct + 1

            # display the confusion matrix
            # and how many results were false and true
            for i in range(26):
                for j in range(26):
                    print "%2.f" %Matrix[i][j],
                print

            print "We made " + str(false) + " false predictions"
            print "We made " + str(correct) + " correct predictions"

            # setting vector support machine up and train using linear kernel
            clf = svm.SVC(kernel = 'poly',gamma = testval)
            print clf.fit(element,LabelSets[x].ravel())
            # confusion matrix
            Matrix = [[0 for i in range(26)] for j in range(26)]
            # prediction counters
            false = 0
            correct = 0

            # testing all the elements in a partition to test the prediction model
            for i in range(len(LabelSets[y])):
                if(int(clf.predict(np.array((testing[y])[i], ndmin=2))[0]) != int((LabelSets[y])[i][0])):
                    Matrix[int(clf.predict(np.array((testing[y])[i], ndmin=2))[0])][int((LabelSets[y])[i][0])] += 1
                    false = false + 1
                if(int(clf.predict(np.array((testing[y])[i], ndmin=2))[0]) == int((LabelSets[y])[i][0])):
                    Matrix[int(clf.predict(np.array((testing[y])[i], ndmin=2))[0])][int((LabelSets[y])[i][0])] += 1
                    correct = correct + 1

            # display the confusion matrix
            # and how many results were false and true
            for i in range(26):
                for j in range(26):
                    print "%2.f" %Matrix[i][j],
                print
            print "We made " + str(false) + " false predictions"
            print "We made " + str(correct) + " correct predictions"



            # setting vector support machine up and train using linear kernel
            clf = svm.SVC(kernel = 'rbf',gamma = testval)
            print clf.fit(element,LabelSets[x].ravel())
            # confusion matrix
            Matrix = [[0 for i in range(26)] for j in range(26)]
            # prediction counters
            false = 0
            correct = 0

            # testing all the elements in a partition to test the prediction model
            for i in range(len(LabelSets[y])):
                if(int(clf.predict(np.array((testing[y])[i], ndmin=2))[0]) != int((LabelSets[y])[i][0])):
                    Matrix[int(clf.predict(np.array((testing[y])[i], ndmin=2))[0])][int((LabelSets[y])[i][0])] += 1
                    false = false + 1
                if(int(clf.predict(np.array((testing[y])[i], ndmin=2))[0]) == int((LabelSets[y])[i][0])):
                    Matrix[int(clf.predict(np.array((testing[y])[i], ndmin=2))[0])][int((LabelSets[y])[i][0])] += 1
                    correct = correct + 1

            # display the confusion matrix
            # and how many results were false and true
            for i in range(26):
                for j in range(26):
                    print "%2.f" %Matrix[i][j],
                print
            print "We made " + str(false) + " false predictions"
            print "We made " + str(correct) + " correct predictions"
