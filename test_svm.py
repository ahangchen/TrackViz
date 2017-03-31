#################################################
# logRegression: Logistic Regression
# Author : zouxy
# Date   : 2014-03-02
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################
import random

import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn


# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
    # return inX

# train a logistic regression model using some optional optimize algorithm
# input: train_x is a mat datatype, each row stands for one sample
#		 train_y is mat datatype too, each row is the corresponding label
#		 opts is optimize option include step and maximum number of iterations


def hinge(x):
    return np.maximum(np.zeros(shape=np.shape(x)), 1 - x)


def trainLogRegres(train_x, train_y, opts):
    # calculate training time
    startTime = time.time()

    numSamples, numFeatures = np.shape(train_x)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = np.random.rand(numFeatures, 1)
    errors = []
    iter_cnt = 0
    # optimize through gradient descent algorilthm
    for k in range(maxIter):
        if opts['optimizeType'] == 'gradDescent':  # gradient descent algorilthm
            output = sigmoid(train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose() * error
            # if np.sum(np.abs(error)) / numSamples < 0.05:
            #     print 'converge on step %d\n' % k
            #     break
        elif opts['optimizeType'] == 'stocGradDescent':  # stochastic gradient descent
            idx = random.randint(0, numSamples - 1)
            output = sigmoid(train_x[idx, :] * weights)
            error = train_y[idx, 0] - output
            weights = weights + alpha * train_x[idx, :].transpose() * error
        elif opts['optimizeType'] == 'smoothStocGradDescent':  # smooth stochastic gradient descent
            # randomly select samples to optimize for reducing cycle fluctuations
            dataIndex = range(numSamples)
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del (dataIndex[randIndex])  # during one interation, delete the optimized sample
        else:
            raise NameError('Not support optimize method type!')
        output = sigmoid(train_x * weights)
        error = train_y - output
        cur_err = np.sum(np.abs(error))/numSamples
        errors.append(cur_err)
        iter_cnt += 1
        if cur_err < 0.1:
            print 'converge on step %d\n' % k
            break

    plt.plot(range(iter_cnt), errors)
    plt.show()
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return weights


def trainSVM(train_x, train_y, opts):
    # calculate training time
    startTime = time.time()

    numSamples, numFeatures = np.shape(train_x)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = np.random.rand(numFeatures, 1)

    # optimize through gradient descent algorilthm
    for k in range(maxIter):
        if opts['optimizeType'] == 'gradDescent':  # gradient descent algorilthm
            if train_x * weights * train_y < 1:
                weights += alpha * (weights - train_x.transpose() * train_y)
            else:
                weights += alpha * weights

        elif opts['optimizeType'] == 'stocGradDescent':  # stochastic gradient descent
            idx = random.randint(0, numSamples - 1)
            if train_x[idx, :] * weights * train_y[idx, 0] < 1:
                weights += alpha * (weights - train_x[idx, :].transpose() * train_y[idx, 0])
            else:
                weights += alpha * weights

        elif opts['optimizeType'] == 'smoothStocGradDescent':  # smooth stochastic gradient descent
            # randomly select samples to optimize for reducing cycle fluctuations
            dataIndex = range(numSamples)
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del (dataIndex[randIndex])  # during one interation, delete the optimized sample
        else:
            raise NameError('Not support optimize method type!')

        error = hinge(train_x * weights)
        cur_err = np.sum(np.abs(error)) / numSamples
        print cur_err
        if cur_err < 0.1:
            print 'converge on step %d\n' % k
            break
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return weights


# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    matchCount = 0
    for i in xrange(numSamples):
        predict = hinge(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# show your trained logistic regression model only available with 2-D data
def showLogRegres(weights, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = np.shape(train_x)
    if numFeatures != 3:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    # weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def loadData():
    train_x = []
    train_y = []
    fileIn = open('testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return np.mat(train_x), np.mat(train_y).transpose()

if __name__ == '__main__':
    print hinge(np.mat([[1.0, 0.5], [0.2, 1.5]]))
    ## step 1: load data
    print "step 1: load data..."
    train_x, train_y = loadData()
    test_x = train_x
    test_y = train_y

    ## step 2: training...
    print "step 2: training..."
    opts = {'alpha': 0.1, 'maxIter': 4000, 'optimizeType': 'stocGradDescent'}
    optimalWeights = trainSVM(train_x, train_y, opts)

    ## step 3: testing
    print "step 3: testing..."
    accuracy = testLogRegres(optimalWeights, test_x, test_y)

    ## step 4: show the result
    print "step 4: show the result..."
    print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
    showLogRegres(optimalWeights, train_x, train_y)