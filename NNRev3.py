# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:47:56 2020

@author: Daniel
"""

import numpy as np
import mysql
import mysql.connector
#=========== Constanta =============#
learn_rate = 0.1
epoch = 1000
momentum = 0.5
inputSize = 36
hiddenSize = 2
outputSize = 1

#===================================#

def AmbilDataMySQL():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database = "dbskripsi"
    )
    mycursor = mydb.cursor()
    try:
        
        sql = "select * from datakarakter"
        
        mycursor.execute(sql)
        records = mycursor.fetchall()
        print(mycursor.rowcount, "record get.")
        rows = mycursor.rowcount
        sql2 = "SELECT count(*)  FROM information_schema.columns WHERE table_name ='datakarakter';"
        mycursor.execute(sql2)
        recCol = mycursor.fetchone()
        column = recCol[0]
        print (column)
        
        matrix = [[0 for x in range(column-1)] for y in range(rows)] 
        indexRow = 0
        for row in records:
            for col in range(0, column-1):
                matrix[indexRow][col] = row[col+1]
            indexRow = indexRow+1            
        
        np_matrix = np.array(matrix)
        print("Shape Awal= ",np_matrix.shape[0],",", np_matrix.shape[1])
        trys = np_matrix[:, 36]      
        np_del = np.delete(np_matrix, 36, 1)
        return np_del, trys
    except:
        print ("Record failed to updated.")
        
def ScalingData(X):
    meanX = np.mean(X)
    stdDev = np.std(X)
    z= (X-meanX)/stdDev
    return z
inputs, outputs = AmbilDataMySQL()
inputs = ScalingData(inputs)
outputs = ScalingData(outputs)
shap = outputs.shape[0]
outputs = np.reshape(np.array(outputs),(shap,1))
bias1 = 0.5
bias2 = 0.7

#==============================Activation Function======================#
def ActivationFunction(houtput):
    X = 1.0 / (1.0 + np.exp(-houtput) )
    return X, houtput
def ActivationDerivative(dA, Z):
    A, Z = ActivationFunction(Z)
    dZ= dA *A *(1 - A)
    return dZ
def SigmoidDerivative(x):
    return x *(1-x)


#===============================Forward Pass ====================================#    
def InitWeight():
    np.random.seed(1) 
    W1 = np.random.randn(inputSize, hiddenSize) 
    W2 = np.random.randn(hiddenSize, outputSize)
    return W1, W2

def LinearForwardPass(data, weight,bias):#input->hidden
    Z = np.dot(data, weight) + bias
    cache = (data, weight, bias)
    return Z, cache

def LinearActivationForward(data,weight,bias): #input->hidden
    Z, linear_cache = LinearForwardPass(data, weight, bias)
    A, activation_cache = ActivationFunction(Z)
    cache = (linear_cache,activation_cache)
    return A, cache

def L_model_forward(X, weight1,weight2, bias1,bias2):
    A= X
    caches= []
    A_prev = A
    A, cache = LinearActivationForward(A_prev, weight1, bias1)
    '''forward part1'''
    #caches.append(cache)
    AL, cache = LinearActivationForward(A, weight2, bias2)
    '''forward part 2'''
    #caches.append(cache)
    #print("A LMF: ",A)
    '''Debug. Cache yang terdapat pada A_prev di LBP merupakan A bukan AL, yang seharusnya masuk.'''
    print ("Shape Forward: ",AL.shape)
    #assert  AL.shape == (1, X.shape[1])
    return AL, A

def CostFunction(outputs, cal_output):
    cost = np.mean(np.square(cal_output-outputs))
    print(cost)
    

#=====================Back Pass============================#   
def updateWeight(weight, backward):
    change = learn_rate*backward
    changeTotal = np.sum(change, axis=0)
    changeTotal = np.reshape(changeTotal, weight.shape)
    print(weight)
    print(changeTotal)
    weight2 = weight - changeTotal
    print("changeTotal:",changeTotal)
    print(weight)
    return weight, weight2
    
def OHBackward(output,expected, weight, outputHidden):
    '''Backward dari output ke hidden layer'''
    dCost = -1*(output-expected)
    dLog = SigmoidDerivative(output)
    Backward = outputHidden.dot(dCost * dLog)
    print(Backward.shape)
    oldWeight , newWeight= updateWeight(weight, Backward)
    return dCost, dLog, oldWeight,newWeight
   
def HIBackward(dCost, dLog, oldWeight, inputs, outputHidden):
    ''''Backward dari Hidden ke Input Layer'''
    print("dcost:",dCost.shape,"\n dlog:", dLog.shape,"\noldWight", oldWeight.shape)
    rumus1 = dCost * dLog * oldWeight   
    
    rumus = rumus1 * SigmoidDerivative(outputHidden) * inputs
    '''
def LinearBackwardPass(data, cache):
    A_prev, W, b = cache
    print("Shape Data:", data.shape)
    print("Shape A_prev",A_prev.shape)
    print(data)
    print("A_prev LBP:",A_prev)

    #data= data.reshape(A_prev.shape)
    m = A_prev.shape[1]
    print(m)
    print (A_prev.shape)
  
    dW = (1/m) * np.dot(data, A_prev.T)
    db = (1/m) * np.sum(data, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, data)
    
    return dA_prev, dW,db

def LinearActivationBackward(data, cache):
    linear_cache, activation_cache = cache
    dZ = ActivationDerivative(data, activation_cache)
    dA_prev , dW, db = LinearBackwardPass(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, y, caches):
    y = y.reshape(AL.shape)
    L = len(caches)
    grads = {}
    dAL = np.divide(AL-y, np.multiply(AL, 1-AL))
    
    
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = LinearActivationBackward(dAL, caches[L-1])
    for l in range(L - 1, 0, -1):
        currentcache = caches[l-1]
        grads["dA"+str(l-1)], grads["dW"+str(l)], grads["db"+str(L)] = LinearActivationBackward(dAL, currentcache)
    return grads
    

def updateWeight(W1, W2, grads, b1,b2):
    W1 = W1 - learn_rate * grads["dW"+str(1)]
    W2 = W2 - learn_rate * grads["dW2"]
    b1 = bias1 - learn_rate* grads["db1"]
    b2 = bias2 - learn_rate *grads["db2"]
    
'''
W1, W2 = InitWeight()

Expected, cache = L_model_forward(inputs, W1,W2, bias1, bias2)



#print(A_prev)
#print (Forward.shape)
CostFunction(outputs, Expected)
dcost, dlog, oldWeight2, newWeight2 = OHBackward(outputs, Expected, W2, cache)
HIBackward(dcost, dlog, oldWeight2, inputs, cache)
''''expected -> kalkulasi program 
outputs -> hasil scalar database'''
#L_model_backward(Forward, outputs, cache)

