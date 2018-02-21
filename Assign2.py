import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def coin():
    return random.randrange(0,2)

def coins20():
    X = []
    for i in range(0,20):
        X.append(coin())
    return X

def mili():
    E = []
    for i in range(0,1000000):
        E.append(coins20())
    return E

mil_reps = mili()
np.shape(mil_reps)
def frqs(bs):
    data = []
    a = 0.5
    while a < 1.05:
        count = 0
        for i in range(0,1000000):
            freq = 0.05*sum(bs[i])
            if freq >= a:
                count = count+1
        data.append(count)
        a = a+0.05
    return data

emp_freqs = frqs(mil_reps)
emp_freqs
aaaa = [a*0.05 for a in range(10,21)]

plt.figure(1)
plt.plot(aaaa, emp_freqs, label="Empirical freq. of observations", color="red", linestyle=':')
plt.title(" 20 coin flips plot of frequency", color="blue")
plt.xlabel("Alpha", color="blue")
plt.ylabel("Frequency greater than alpha", color="blue")
plt.legend()
plt.show()

def hoff(es):
    hoff_bound = []
    for e in es:
        hoff_bound.append(np.exp(-2*20*(e-0.5)**2)*1000000)
    return hoff_bound

def markov(aaaa):
    mark_bound=[]
    for a in aaaa:
        mark_bound.append(1000000*0.5/a);
    return mark_bound

plt.figure(2)
plt.plot(aaaa, emp_freqs, label="Empirical frequency", color="red", linestyle=':')
plt.plot(aaaa, hoff(aaaa), label = "Hoeffding's Bound", color="blue", linestyle=':')
plt.plot(aaaa, markov(aaaa), label = "Markov's Bound", color="green", linestyle=':')
plt.title("Hoeffding's and Markov's Bounds", color="blue")
plt.xlabel(" alpha", color="red")
plt.ylabel("Frequency of average greater than alpha", color="red")
plt.legend()
plt.axis([0.5,1.0, 0,1000000])
#plt.legend(bbox_to_anchor=(0, -0.15, 1, 0), loc=2, borderaxespad=0)
plt.legend()
plt.show()

#Handling Data
locationTrain = 'IrisTrainML.dt'
locationTest = 'IrisTestML.dt'
dfTrain = pd.read_csv(locationTrain, header=None, sep=' ')
dfTest = pd.read_csv(locationTest, header=None, sep=' ')

dfTrainNO2 = dfTrain.loc[dfTrain[2] != 2]
dfTestNO2 = dfTest.loc[dfTest[2] != 2]

irisTrainX = dfTrainNO2.as_matrix(columns=[0,1])
irisTrainY = dfTrainNO2.as_matrix(columns=[2])
irisTestX = dfTestNO2.as_matrix(columns=[0,1])
irisTestY = dfTestNO2.as_matrix(columns=[2])

np.place(irisTrainY, irisTrainY == 0, -1)
np.place(irisTestY, irisTestY == 0, -1)


def bigX(inputData):
    if inputData.shape[0] > inputData.shape[1]:
        inputNoTilde = inputData
    else:
        inputNoTilde = np.transpose(inputData)
    ones = np.ones((len(inputNoTilde), 1), dtype=np.int)
    inputX = np.hstack((inputNoTilde, ones))
    return inputX


# Weights
def initWeights(data):
    dimensions = data.shape[1]
    weights = np.zeros((dimensions))
    return weights


# Simple Gradient
def gradOneSimple(x, y, w):
    yMULTx = np.multiply(y,x)
    wDOTx = np.dot(np.transpose(w),x)
    yMULTwtx = np.multiply(y, wDOTx)
    exp = np.exp(yMULTwtx)
    gradient = np.divide(yMULTx, 1 + exp)
    return gradient


# Full Gradient
def gradient(dataX, dataY, weights):
    accum = initWeights(dataX)
    n = len(dataY)
    for i in range(len(dataX)):
        gradient = gradOneSimple(dataX[i], dataY[i], weights)
        accum += gradient
    mean = np.divide(accum, n)
    gradient = np.negative(mean)
    return gradient


# Function for updating weights.
def updateWeights(oldWeights, direction, learningRate):
    newWeight = oldWeights + np.multiply(learningRate, direction)
    return newWeight


# Logistic regression model. Implementation of LFD algorithm.
def logReg(dataX, dataY, learningRate):
    X = bigX(dataX)
    weights = initWeights(X)
    for i in range(0,1000):
        g = gradient(X, dataY, weights)
        direction = -g
        weights = updateWeights(weights, direction, learningRate)
    return weights


# Building affine linear model and reporting results.
vectorWandB = logReg(irisTrainX, irisTrainY, 0.1)

vectorW = np.transpose(vectorWandB[:-1])

b = vectorWandB[-1]

print(vectorWandB)
print('\n')
print('Affine linear model build. w: ' + str(vectorW) + '   b: '
                                             + str(b) + '\n')


# Function that computes conditional probability using logistic regression.
def conditionalProb(x, vectorW, b):
    wDOTx = np.dot(vectorW, x)
    y = wDOTx + b
    exp = np.exp(y)
    prob = np.divide(exp, 1 + exp)
    return prob


# Function that classifies using the probability from logistic regression.
def linearClassifier(x, vectorW, b ):
    y = -2
    prob = conditionalProb(x, vectorW, b)
    if prob > 0.5:
        y = 1
    else:
        y = -1
    return y


# Function that finds the number of wrong classifications for test data.
def testingFalse(trainX, trainY, testX, testY, learningRate):
    trueCount = 0
    falseCount = 0
    weights = logReg(trainX, trainY, learningRate)
    vectorW = weights[:-1]
    b = weights[-1]
    for i in range(0,len(testY)):
        if linearClassifier(testX[i], vectorW, b) == testY.item(i):
            trueCount += 1
        else:
            falseCount += 1
    return falseCount


# Finding the empirical loss of linear classification model with a zero-one loss
# function.
def zeroOneLoss(trainX, trainY, testX, testY, learningRate):
        N = len(testY)
        misClassified = testingFalse(trainX, trainY, testX, testY, learningRate)
        zeroOneLoss = (1/N)*misClassified
        return zeroOneLoss

lossTrain = zeroOneLoss(irisTrainX, irisTrainY, irisTrainX, irisTrainY, 0.1)
lossTest = zeroOneLoss(irisTrainX, irisTrainY, irisTestX, irisTestY, 0.1)
print("0-1 loss of logistic regression linear classifier on training data: " + str(lossTrain))
print("0-1 loss of logistic regression linear classifier on testing data: " + str(lossTest))


