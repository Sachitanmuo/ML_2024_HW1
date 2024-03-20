import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import math


def sig(a):
    return 1.0 / (1 + np.exp(-a))


def phi(i, j):
    if j == 0:
        return 1.0
    return sig((df.x[i] - 3 * j / m) / 0.1)


def phifunc(j, x):
    if j == 0:
        return 1.0
    return sig((x - 3 * j / m) / 0.1)


def getans(design):
    return np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(np.transpose(design), design)), np.transpose(design)
        ),
        df.t[:50],
    )


def gety(m, x, flag):
    design = np.zeros((50, m))
    for i in range(50):
        for j in range(m):
            design[i][j] = phi(i, j)
    d = 0
    for i in range(m):
        d += phifunc(i, x) * getans(design)[i]
    if m == 1 and flag == False:
        d = np.full((1000, 1), d)
    return d


##1-2


def totalerror(m):
    s = 0
    design = np.zeros((50, m))
    for i in range(50):
        for j in range(m):
            design[i][j] = phi(i, j)
    y = np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(np.transpose(design), design)), np.transpose(design)
        ),
        df.t[:50],
    )
    y = np.matmul(design, y)
    for i in range(50):
        s += (df.t[i] - y[i]) ** 2
    return s * 0.5


def totaltesterror(m, n):
    design = np.zeros((50, m))
    for i in range(50):
        for j in range(m):
            design[i][j] = phi(i, j)
    y = np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(np.transpose(design), design)), np.transpose(design)
        ),
        df.t[:50],
    )
    k = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            k[i][j] = phifunc(i, df.x[50 + j])
    f = 0
    k = np.matmul(np.transpose(y), k)
    for j in range(n):
        f += (df.t[j + 50] - k[j]) ** 2
    return f * 0.5


# 1-3
def fivefolderror(m, x, t, xtest, ttest):
    design = np.zeros((40, m))
    for i in range(40):
        for j in range(m):
            design[i][j] = phifunc(j, x[i])
    y = np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(np.transpose(design), design)), np.transpose(design)
        ),
        t,
    )
    k = np.matmul(design, y)
    f = 0
    for j in range(40):
        f += (t[j] - k[j]) ** 2
    g = np.zeros(2)
    g[0] = f * 0.5
    ##test
    k1 = np.zeros((m, 10))
    for i in range(m):
        for j in range(10):
            k1[i][j] = phifunc(i, xtest[j])
    f1 = 0
    k1 = np.matmul(np.transpose(y), k1)
    for j in range(10):
        f1 += (ttest[j] - k1[j]) ** 2
    g[1] = f1 * 0.5
    # print(g[1])
    return g


def get5ans(design, t):
    return np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(np.transpose(design), design)), np.transpose(design)
        ),
        t,
    )


def get5y(m, x, flag, t):
    design = np.zeros((40, m))
    for i in range(40):
        for j in range(m):
            design[i][j] = phi(i, j)
    d = 0
    for i in range(m):
        d += phifunc(i, x) * get5ans(design, t)[i]
    if m == 1 and flag == False:
        d = np.full((1000, 1), d)
    return d


# 1-4
def getregans(design, m, l):
    return np.matmul(
        np.matmul(
            np.linalg.inv(
                l * np.eye(m, dtype=int) + np.matmul(np.transpose(design), design)
            ),
            np.transpose(design),
        ),
        df.t[:50],
    )


def getregy(m, x, l):
    design = np.zeros((50, m))
    for i in range(50):
        for j in range(m):
            design[i][j] = phi(i, j)
    d = 0
    for i in range(m):
        d += phifunc(i, x) * getregans(design, m, l)[i]
    if m == 1:
        d = np.full((1000, 1), d)
    return d


def regtotalerror(m):
    s = 0
    design = np.zeros((50, m))
    for i in range(50):
        for j in range(m):
            design[i][j] = phi(i, j)
    y = np.matmul(
        np.matmul(
            np.linalg.inv(
                0.1 * np.eye(m, dtype=int) + np.matmul(np.transpose(design), design)
            ),
            np.transpose(design),
        ),
        df.t[:50],
    )
    f = y
    y = np.matmul(design, y)
    for i in range(50):
        s += (df.t[i] - y[i]) ** 2
    return s * 0.5 + 0.5 * 0.1 * np.linalg.norm(f) ** 2


def regtotal1error(m):
    s = 0
    design = np.zeros((50, m))
    for i in range(50):
        for j in range(m):
            design[i][j] = phi(i, j)
    y = np.matmul(
        np.matmul(
            np.linalg.inv(
                1 * np.eye(m, dtype=int) + np.matmul(np.transpose(design), design)
            ),
            np.transpose(design),
        ),
        df.t[:50],
    )
    f = y
    y = np.matmul(design, y)
    for i in range(50):
        s += (df.t[i] - y[i]) ** 2
    return s * 0.5 + 0.5 * 1 * np.linalg.norm(f) ** 2


def regtotal2error(m):
    s = 0
    design = np.zeros((50, m))
    for i in range(50):
        for j in range(m):
            design[i][j] = phi(i, j)
    y = np.matmul(
        np.matmul(
            np.linalg.inv(
                0.01 * np.eye(m, dtype=int) + np.matmul(np.transpose(design), design)
            ),
            np.transpose(design),
        ),
        df.t[:50],
    )
    f = y
    y = np.matmul(design, y)
    for i in range(50):
        s += (df.t[i] - y[i]) ** 2
    return s * 0.5 + 0.5 * 0.01 * np.linalg.norm(f) ** 2


def regtotaltesterror(m, n):
    design = np.zeros((50, m))
    for i in range(50):
        for j in range(m):
            design[i][j] = phi(i, j)
    y = np.matmul(
        np.matmul(
            np.linalg.inv(
                0.1 * np.eye(m, dtype=int) + np.matmul(np.transpose(design), design)
            ),
            np.transpose(design),
        ),
        df.t[:50],
    )
    k = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            k[i][j] = phifunc(i, df.x[50 + j])
    f = 0
    k = np.matmul(np.transpose(y), k)
    for j in range(n):
        f += (df.t[j + 50] - k[j]) ** 2
    return f * 0.5 + 0.05 * np.linalg.norm(y) ** 2


# 2
def geterror(design, x):
    f = np.zeros((10, 1000))
    for i in range(10):
        f[i] = phifunc(i, x)
    return 1 + np.matmul(np.transpose(f), np.matmul(getsn(design), f))


def getmy(m, x, y):
    d = np.zeros((10, 1000))
    for i in range(m):
        d[i] = phifunc(i, x)

    return np.matmul(y, d)


def getsn(design):
    return np.linalg.inv(
        1e-6 * np.eye(m, dtype=int) + np.matmul(np.transpose(design), design)
    )


def getmean(design, t):
    return np.matmul(getsn(design), np.matmul(np.transpose(design), t))


def getdesign(n, m):
    design = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            design[i][j] = phi(i, j)
    return design


# main code

data = pd.read_csv(r"Hw1.csv")
df = pd.DataFrame(data, columns=["x", "t"])
plt.scatter(df.x[:50], df.t[:50])
plt.show()
m = 20
marray = [1, 3, 5, 10, 20, 30]

# 1-1
for i in range(6):
    m = marray[i]
    x = np.linspace(0, 3, 1000)
    plt.scatter(df.x[:50], df.t[:50])
    plt.plot(x, gety(m, x, 0))
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("T")
    plt.title("Training Set and Fitting Curve for M=" + str(m))
    plt.show()

##1-2
trainerror = np.zeros((30, 2))
for i in range(30):
    trainerror[i][0] = i + 1
for i in range(30):
    m = i
    trainerror[i][1] = totalerror(i)
trainerror = np.transpose(trainerror)
plt.bar(trainerror[0] - 0.2, trainerror[1], 0.4, label="training")


plt.xlabel("M")
plt.ylabel("Error")
# plt.show()
##test
testerror = np.zeros((30, 2))
for i in range(30):
    testerror[i][0] = i + 1
for i in range(30):
    m = i
    testerror[i][1] = totaltesterror(i, 20)
testerror = np.transpose(testerror)
plt.bar(testerror[0] + 0.2, testerror[1], 0.4, label="testing")
plt.grid(True)
plt.title("MSE")
plt.xlabel("M")
plt.ylabel("Error")
plt.legend()
plt.show()

# 1-3


xtrain = np.zeros((5, 40))
ttrain = np.zeros((5, 40))
xtest = np.zeros((5, 10))
ttest = np.zeros((5, 10))
xtrain[0] = df.x[10:50]
xtrain[1] = np.concatenate((df.x[:10], df.x[20:50]))
xtrain[2] = np.concatenate((df.x[:20], df.x[30:50]))
xtrain[3] = np.concatenate((df.x[:30], df.x[40:50]))
xtrain[4] = df.x[:40]
xtest[0] = df.x[:10]
xtest[1] = df.x[10:20]
xtest[2] = df.x[20:30]
xtest[3] = df.x[30:40]
xtest[4] = df.x[40:50]
ttrain[0] = df.t[10:50]
ttrain[1] = np.concatenate((df.t[:10], df.t[20:50]))
ttrain[2] = np.concatenate((df.t[:20], df.t[30:50]))
ttrain[3] = np.concatenate((df.t[:30], df.t[40:50]))
ttrain[4] = df.t[:40]
ttest[0] = df.t[:10]
ttest[1] = df.t[10:20]
ttest[2] = df.t[20:30]
ttest[3] = df.t[30:40]
ttest[4] = df.t[40:50]
errorrecord = np.zeros((22, 2))
xx = np.arange(1, 23, 1)

for k in range(5):
    trainerror = np.zeros((22, 2))
    for i in range(22):
        m = i
        trainerror[i] = fivefolderror(i, xtrain[k], ttrain[k], xtest[k], ttest[k])
        errorrecord[i] += trainerror[i]
    trainerror = np.transpose(trainerror)
    #    print(trainerror)

    plt.show()
    plt.bar(xx - 0.2, trainerror[:][0], 0.4, label="training error")
    plt.bar(xx + 0.2, trainerror[:][1], 0.4, label="testing error")
    plt.title("Repetition " + str(k + 1) + " of Training and Testing Error")
    plt.legend()
    plt.show()
m = 15
errorrecord = np.transpose(errorrecord)
print(errorrecord)
print(errorrecord[0] + errorrecord[1])
plt.bar(xx - 0.2, errorrecord[:][0], 0.4, label="training error")
plt.bar(xx + 0.2, errorrecord[:][1], 0.4, label="testing error")
plt.title("total MSE error")
plt.legend()
plt.show()
plt.bar(xx, errorrecord[:][1] + errorrecord[:][0], 0.4)
plt.title("total training+testing MSE error")
plt.show()
for k in range(5):
    x = np.linspace(0, 3, 1000)
    plt.plot(x, get5y(m, x, 0, ttrain[k]))
    plt.scatter(xtrain[k], ttrain[k], label="train")
    plt.scatter(xtest[k], ttest[k], label="test")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("T")
    plt.title(
        "Using Data Points " + str(k * 10) + "~" + str(k * 10 + 10) + " To Plot Graph"
    )
    plt.legend()
    plt.show()


# 1-4
# 1-4-1
marray = [1, 3, 5, 10, 20, 30]
for i in range(6):
    m = marray[i]
    plt.scatter(df.x[:50], df.t[:50])
    x = np.linspace(0, 3, 1000)
    plt.plot(x, getregy(m, x, 0.1), label="lambda=0.1")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("T")
    plt.title("Training Set and Regularized Fitting Curve for M=" + str(m))
    plt.legend()
    plt.show()
    plt.plot(x, getregy(m, x, 0.1), label="lambda=0.1")
    plt.plot(x, getregy(m, x, 1), label="lambda=1")
    plt.plot(x, getregy(m, x, 0.01), label="lambda=0.01")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("T")
    plt.title(
        "Training Set and Regularized Fitting Curve for M="
        + str(m)
        + " with different lambda"
    )
    plt.legend()
    plt.show()

##1-4-2
trainerror = np.zeros((30, 2))
for i in range(30):
    trainerror[i][0] = i + 1
for i in range(30):
    m = i
    trainerror[i][1] = regtotalerror(i)
trainerror = np.transpose(trainerror)
plt.bar(trainerror[0] - 0.2, trainerror[1], 0.4, label="training set")
# plt.show()
##test
testerror = np.zeros((30, 2))
for i in range(30):
    testerror[i][0] = i + 1
for i in range(30):
    m = i
    testerror[i][1] = regtotaltesterror(i, 20)
testerror = np.transpose(testerror)
plt.bar(testerror[0] + 0.2, testerror[1], 0.4, label="testing set")
plt.grid(True)
plt.title("Regularized MSE")
plt.xlabel("M")
plt.ylabel("Error")
plt.legend()
plt.show()


plt.show()
m = 10
narray = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
for f in range(10):
    n = narray[f]
    plt.scatter(df.x[:n], df.t[:n])
    y = getdesign(n, m)
    des = y
    y = getmean(y, df.t[:n])
    x = np.linspace(0, 3, 1000)
    plt.plot(x, getmy(m, x, y))
    e = geterror(des, x)
    error = np.zeros(1000)
    for i in range(1000):
        error[i] = math.sqrt(e[i][i])
    plt.fill_between(x, getmy(m, x, y) + error, getmy(m, x, y) - error, alpha=0.3)
    plt.grid(True)
    plt.xlim(0, 3)
    if n < 11:
        plt.ylim(-50, 50)
    plt.title("Bayesian Linear Regression with " + str(n) + " training data")
    plt.xlabel("X")
    plt.ylabel("T")
    plt.show()
