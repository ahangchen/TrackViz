import random

import matplotlib.pyplot as plt


# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat


def quad_grad(w, train):
    x = train[: -1]
    y = train[-1]
    return [x[i] ** 2 * w - x[i] * y for i in range(len(x))]


# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    errors = []
    for epoch in range(n_epoch):
        sum_error = 0
        idx = random.randint(0, len(train) - 1)
        row = train[idx]
        yhat = predict(row, coef)
        error = yhat - row[-1]
        sum_error += error ** 2
        coef[0] -= l_rate * (quad_grad(coef[0], row)[0])
        for i in range(len(row) - 1):
            coef[i + 1] -= l_rate * (quad_grad(coef[i+1], row)[0])
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        errors.append(sum_error)
    plt.plot(range(n_epoch), errors)
    plt.show()
    return coef


def coefficients_gd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    errors = []
    for epoch in range(n_epoch):
        idx = random.randint(0, len(train) - 1)
        row = train[idx]
        yhat = predict(row, coef)
        error = yhat - row[-1]
        sum_error = error ** 2
        coef[0] -= l_rate * sum([quad_grad(coef[0], tmp_train)[0] for tmp_train in train])/len(train)
        for i in range(len(row) - 1):
            coef[i + 1] -= l_rate * sum([quad_grad(coef[i+1], tmp_train)[0] for tmp_train in train])/len(train)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        errors.append(sum_error)
    plt.plot(range(n_epoch), errors)
    plt.show()
    return coef

# Calculate coefficients
X = [(i + random.uniform(-3, 3)) for i in range(50)]
Y = [0.7 * X[i] + random.uniform(-5, 5) for i in range(50)]
dataset = [[X[i], Y[i]] for i in range(len(X))]
l_rate = 0.001
n_epoch = 40
coef = coefficients_sgd(dataset, l_rate, n_epoch)
plt.figure()
plt.plot(X, Y, linestyle='', marker='.')
yhyp = [predict([X[i], Y[i]], coef) for i in range(len(X))]
plt.plot(X, yhyp, linestyle='-')
plt.show()
print(coef)
