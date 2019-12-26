# from numpy import *
# import os
#
# a = 2
# print(a.__pow__(2))
#
# a = matrix([[-2,3],[2,5]])
# print(exp(a),a+a,2*a)
#
# a[a>0] = 0
# print('new a = ',a)
#
# a = matrix([[1,2,3],[2,3,1]])
# b = conj(a)
# print('conj(a) shape = ',b.shape)

import numpy as np

a = np.mat([[2, 5, 7, 8, 9, 89], [6, 7, 5, 4, 6, 4]])

def getPositon():
    a = np.mat([[2, 5, 7, 8, 9, 89], [6, 7, 5, 4, 6, 4]])

    raw, column = a.shape# get the matrix of a raw and column

    _positon = np.argmax(a)# get the index of max in the a
    print(_positon)
    m, n = divmod(_positon, column)
    print("The raw is " ,m)
    print("The column is ",  n)
    print("The max of the a is ", a[m , n])
    print(a.shape[0])
getPositon()
