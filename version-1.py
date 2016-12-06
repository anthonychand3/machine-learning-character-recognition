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

a = b = c = d = e = f = g = h = ic = j = k = l = m = n = o = p = q = r = s = t = u = v = w = x = y = z = 0

# Iterate through all of the elements in the testing data
for i in range(len(labels)):

    # if the prediction doesnt equal the actual label then it is false
    if(chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A'))) != chr(int(labels[i][0] + ord('A')))):
        #print "This is the False prediction: " + chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A')))
        #print "This is the correct answer: " + chr(int(labels[i][0] + ord('A')))
        false = false + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'A'):
            a = a + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'B'):
            b = b + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'C'):
            c = c + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'D'):
            d = d + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'E'):
            e = e + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'F'):
            f = f + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'G'):
            g = g + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'H'):
            h = h + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'I'):
            ic = ic + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'J'):
            j = j + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'K'):
            k = k + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'L'):
            l = l + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'M'):
            m = m + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'N'):
            n = n + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'O'):
            o = o + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'P'):
            p = p + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'Q'):
            q = q + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'R'):
            r = r + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'S'):
            s = s + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'T'):
            t = t + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'U'):
            u = u + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'V'):
            v = v + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'W'):
            w = w + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'X'):
            x = x + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'Y'):
            y = y + 1

        if(chr(int(testData[i][0] + ord('A'))) == 'Z'):
            z = z + 1


    # if the prediction equals the label then it is a correct prediction
    if(chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A'))) == chr(int(labels[i][0] + ord('A')))):
        #print "This is the correct prediction: " + chr(int(clf.predict(np.array(testData[i], ndmin=2))[0] + ord('A')))
        #print "This is the correct answer: " + chr(int(labels[i][0] + ord('A')))
        correct = correct + 1

print "\n"
print "We made " + str(false) + " false predictions"

print "We made " + str(correct) + " correct predictions"

print "A = " + str(a)

print "B = " + str(b)

print "C = " + str(c)

print "D = " + str(d)

print "E = " + str(e)

print "F = " + str(f)

print "G = " + str(g)

print "H = " + str(h)

print "I = " + str(ic)

print "J = " + str(j)

print "K = " + str(k)

print "L = " + str(l)

print "M = " + str(m)

print "N = " + str(n)

print "O = " + str(o)

print "P = " + str(p)

print "Q = " + str(q)

print "R = " + str(r)

print "S = " + str(s)

print "T = " + str(t)

print "U = " + str(u)

print "V = " + str(v)

print "W = " + str(w)

print "X = " + str(x)

print "Y = " + str(y)

print "Z = " + str(z)
