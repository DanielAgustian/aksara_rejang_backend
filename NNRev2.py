# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:44:48 2020

@author: Daniel
"""
import numpy as np
import mysql
import mysql.connector
import pyrebase

learning_rate = 0.091
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
    return z, meanX, stdDev

def ScalingDataX(X):
    a= X.shape
    print(a[1])
    print(X.shape)
    meanTemp = [0.0] * a[1]
    stdDevTemp = [0.0]*a[1]
    for j in range(0, a[1]-1):
        temp = [0.0] * a[0]
        for i in range(0, a[0]):
            temp [i] = X[i][j]
        meanTemp[j] = np.mean(temp)
        stdDevTemp[j] = np.std(temp)
        
    for j in range(0, a[1]-1):
        for i in range(0, a[0]):
            X[i][j] = (X[i][j]-meanTemp[j])/stdDevTemp[j]
    print(X.shape)
    return X, meanTemp, stdDevTemp
    
def Pyrebase(w1,w2, mean, std, mean2, std2):
    config = {    
        "apiKey": "AIzaSyCzzLFw1jpREv_DLdAdStDzmLHWJTESXyw",
        "authDomain": "skripsi-478d0.firebaseapp.com",
        "databaseURL": "https://skripsi-478d0.firebaseio.com",
        "projectId": "skripsi-478d0",
        "storageBucket": "skripsi-478d0.appspot.com",
        "messagingSenderId": "909004760048",
        "appId": "1:909004760048:web:991f711cd5e94fab89f335",
        "measurementId": "G-9MSPP71RSY" 
    }

    firebase = pyrebase.initialize_app(config)
    auth = firebase.auth()
    email = "danielagustian32160025@gmail.com"
    password = "Brawijaya55"
    try:
        #user = auth.create_user_with_email_and_password(email,password)
        signin = auth.sign_in_with_email_and_password(email,password)
        #auth.send_email_verification(signin['idToken'])
        print("Success Sign In User")
    except: 
        print("Failed Create User")
    db = firebase.database()
    w1 = w1.tolist()
    w2 = w2.tolist()
    try:    
        db.child("skripsi").child("w1").set(w1)#push data to generated new key
        db.child("skripsi").child("w2").set(w2)
        db.child("skripsi").child("Data").child("Mean").set(mean)
        db.child("skripsi").child("Data").child("STD").set(std)
        db.child("skripsi").child("Data").child("MeanY").set(mean2)
        db.child("skripsi").child("Data").child("StdY").set(std2)
        #db.child("users").child("OwnKey").set(data)# push data to ourself key.
        #db.child("users").child("OwnKey").update(data) #to update data
        
        print("Data Pushed.")
        #users = db.child("users").get()
        #print(users.val())
        
    except:
        print("Error")
X, y = AmbilDataMySQL()
print ("AmbilDataMySQL Done")
X, meanX, stdDev= ScalingDataX(X)



#X, meanX, stdDev = ScalingData(X)
print("Shape X:",X.shape)
y, meanY, stdDev2 = ScalingData(y)
shap = y.shape[0]
y = np.reshape(np.array(y),(shap,1))
print(y.shape)

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 36
        self.outputSize = 1
        self.hiddenSize = 37
    def initialization_weight(self):
        np.random.seed(1)
        self.W1 = np.random.uniform(low = 0, high = 1, size= (self.inputSize, self.hiddenSize))
        self.W2 = np.random.uniform(low = 0, high = 1, size= (self.hiddenSize, self.outputSize))
        #self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (36x37) weight matrix from input to hidden layer
        #self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (37x1) weight matrix from hidden to output layer
    
    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1)              # dot product of X (25,36)(input) and first set of 36x37 weights
       
        self.z2 = self.sigmoid(self.z)           # activation function
        
        self.z3 = np.dot(self.z2, self.W2)            # dot product of hidden layer (25,37) (z2) and second set of 3x1 (37,1)weights
        
        o = self.sigmoid(self.z3)                   # final activation function
       
        #print("==============Forward Pass End====================")
        return o 
    
    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
    #derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, X, y, o):
    # backward propgate through the network
        self.o_error = y - o                                                  # error in output
        
        self.o_delta = self.o_error*self.sigmoidPrime(o)            # applying derivative of sigmoid to error
        
        self.z2_error = self.o_delta.dot(self.W2.T)                      # z2 error: o_delta: (25,1) & W2 (37,1)
                                                                       #how much our hidden layer weights contributed to output error
        
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)       # z2_error(25,37) & z2 (25,37)
                                                                        # applying derivative of sigmoid to z2 error
    
        
        self.W1 += X.T.dot(self.z2_delta*learning_rate)           #X.T (36,25) & z2_delta(25,37)
                                                                    # adjusting first set (input --> hidden) weights
        
        self.W2 += self.z2.T.dot(self.o_delta*learning_rate)      #z2(25,37) & o_delta(25,1)
                                                                # adjusting second set (hidden --> output) weights
        
        return self.W1, self.W2
    def train (self, X, y):
        o = self.forward(X)
        weight1, weight2 =self.backward(X, y, o)
        return weight1,weight2
        
NN = Neural_Network()
NN.initialization_weight()
epoch = 50000+1
for i in range(0,epoch): # trains the NN 1,000 times
  w1, w2 = NN.train(X, y)
  
  if i % 2000==0  :
      print ("Loss di iterasi ke"+str(i)+": \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
      
  if i == epoch-1:
      print ("Loss di iterasi ke"+str(i)+": \n" + str(np.mean(np.square(y - NN.forward(X)))))
      result = NN.forward(X)
      
count=0     
al = y.shape

print("AL"+str(al[0]))
for i in range(0,al[0]):
    #test = y[i,0]-result[i,0]
    
    y[i,0]= (y[i,0]*stdDev2)+meanY
    result[i,0] = (result[i,0]*stdDev2)+meanY
    
    #print("Testing: "+str(test))
    print("Data ke-"+str(i)+"="+str(y[i,0])+"||"+str(result[i,0])) 
    if(round(y[i,0]== round(result[i,0]) )):
        count = count+1
        #print("Correct")
    
    '''
    if test< 0 :
        test = test*-1
    if test<0.40:
        count= count +1
        #print("CORRECT")
    #print("\n")    
        '''
print("Jumlah:",count)        
print(count/377)
print(w1.shape)
print(w2.shape)
Pyrebase(w1, w2,meanX, stdDev, meanY, stdDev2)
