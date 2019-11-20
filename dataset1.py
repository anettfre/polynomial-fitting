#!/usr/bin/env python3

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

n = 30
start = -2
stop = 2
x = np.linspace(start,stop,n)
eps = 1
np.random.seed(1)
e = np.random.random((n)) * eps
y = x*(np.cos(e+0.5*x**3)+np.sin(0.5*x**3))
plt.plot(x,y,'o')

m = 3
# m = 8
A = np.zeros((n,m)) #making a matrix with the right size

for i in range(m): #this is rows
    for j in range(n): #this is columns
        A[j][i] = (x[j])**i #changing at eatch place from 0 to rigth number


# QR-factorisation
q, r = np.linalg.qr(A) #this gives q and R1

def cholesky(B):
    """Facorise and return av lower triangular matrix
    """
    lengthB = len(B)
    L = np.zeros((lengthB, lengthB))
    D = np.zeros((lengthB, lengthB))
    new_A = B
    for k in range(lengthB):
        L[:, k] = new_A[:, k] / new_A[k][k]
        new_L = L[:, k].reshape(1,lengthB)
        D[k][k] = new_A[k][k]
        tmp_L = np.dot(new_L.T, new_L)
        tmp_Matrix = np.dot(D[k][k], tmp_L)
        # L[:,k] = new_L
        new_A = new_A - tmp_Matrix
        # L[:,k] = new_A[:,k]/abs(new_A[k][k])
        L[:,k] = new_A[:,k]
    new_D = np.sqrt(D)
    R = np.dot(L, new_D)
    return R

def forward_subsitution(R, b):
    lengthR = len(R)
    z = np.zeros(lengthR)
    sum = 0
    for i in range(0, lengthR):
        for j in range(0,i):
            sum += (R[i][j]*z[j])

        z[i] = (b[i] - sum) / R[i][i]
    return z

def back_substitution(R1, bhat):
    lengthr = len(R1)
    l = lengthr-1 #since python start counting at 0
    sum = 0
    # bhat = np.dot(q.T,y)
    z = np.zeros(lengthr)
    z[l] = bhat[l]/R1[l][l]
    i = m-1
    while i >= 0:
        sum = 0
        for j in range(i+1,l+1):
            sum += R1[i][j]*z[j]
        z[i] = (bhat[i] - sum)/R1[i][i]
        i -= 1
    return z

def polynominals(x, z):
    polynom = 0
    for j in range(m):
        polynom += z[j]*x**(j)
    return polynom

CholeskyB = np.dot(A.T, A)
#finding the Condition numbers
condA = np.linalg.cond(A)
condB = np.linalg.cond(CholeskyB)
print("Condition number of A is: ", condA, "Condition number of B is: ", condB)
#calling cholesky and doing forward and backsubstitution
CholeskyR = cholesky(CholeskyB)
CholeskyX = np.dot(A.T, y)
CholeskyForwSub = forward_subsitution(CholeskyR, CholeskyX)
new_Z = back_substitution(CholeskyR.T, CholeskyForwSub)
#finding the polynom
CholeskyP = polynominals(x, new_Z)
#using backsubstitution on r from QR factorisation then finding the polynom
z = back_substitution(r, np.dot(q.T,y))
p = polynominals(x,z)
#plotting both task 1 and 2
plt.plot(x, CholeskyP)
plt.plot(x, p)
plt.title("data set 1 m=3")
plt.savefig('dataset1_m=3.pdf')
plt.show()
