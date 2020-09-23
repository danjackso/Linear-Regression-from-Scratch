###Linear Regression from Scratch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

dataSet = pd.read_csv('Salary_Data.csv')
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values


def hypothesis(theta0, theta1, x):
    return theta0 + (theta1 * x)


def cost(theta0, theta1, X, y):
    costValue = 0
    for (xi, yi) in zip(X, y):
        costValue += 0.5 * ((hypothesis(theta0, theta1, xi) - yi) ** 2)
    return costValue


def derivatives(theta0, theta1, X, y):
    dtheta0 = 0
    dtheta1 = 0
    for (xi, yi) in zip(X, y):
        dtheta0 += hypothesis(theta0, theta1, xi) - yi
        dtheta1 += (hypothesis(theta0, theta1, xi) - yi) * xi
    dtheta0 /= len(X)
    dtheta1 /= len(X)
    return dtheta0, dtheta1


def updateParameters(theta0, theta1, X, y, alpha):
    dtheta0, dtheta1 = derivatives(theta0, theta1, X, y)
    theta0 = theta0 - (alpha * dtheta0)
    theta1 = theta1 - (alpha * dtheta1)
    return theta0, theta1


def RSquared(theta0,theta1,X,y):
    sumSquares = 0
    sumResiduals = 0

    for i in range(0,len(X)):
        yplot = theta0 + theta1 * X[i]

        sumSquares += (y[i]-np.mean(y)) ** 2
        sumResiduals += (y[i]-yplot) **2
    score=1-(sumResiduals/sumSquares)
    return float(score)


def plotLine(theta0, theta1, X, y,iteration):
    plt.close()
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    max_x = np.max(X) + 5
    min_x = np.min(X) - 5

    xplot = np.linspace(min_x, max_x, 1000)
    yplot = theta0 + theta1 * xplot

    score=round((RSquared(theta0,theta1,X,y)),3)
    printOutR='R-Squared: {}'.format(str(score))
    printOutIter='Iteration: {}'.format((str(iteration)))

    ax.text(np.max(X)*0.8,30000,printOutIter)
    ax.text(np.max(X)*0.8,20000,printOutR)

    ax.plot(xplot, yplot, color='g', label='Regression Line')
    ax.scatter(X, y)
    plt.xlim([0, np.max(X) * 1.1])
    plt.ylim([0, np.max(y) * 1.1])
    plt.savefig("value{}.png".format(iteration))
    plt.pause(1e-09)
    plt.show()


def LinearRegression(X, y):
    theta0 = np.random.rand()
    theta1 = np.random.rand()

    tol = 1e-04
    error = 1
    i = 0

    while error > tol:
        errorOld = RSquared(theta0,theta1,X,y)
        i += 1
        if i % 10 == 0:
            plotLine(theta0, theta1, X, y,i)
            if i==500:
                sys.exit()

        theta0, theta1 = updateParameters(theta0, theta1, X, y, 0.005)
        error = abs((RSquared(theta0,theta1,X,y) - errorOld) / errorOld)

    print('Completed')
    print('Iterations: {}'.format(int(i)))
    print('y = {} X + {}'.format(round(float(theta0),3),round(float(theta1),3)))
    print('R-squared:  {}'.format(round(RSquared(theta0,theta1,X,y),3)))

LinearRegression(X, y)
