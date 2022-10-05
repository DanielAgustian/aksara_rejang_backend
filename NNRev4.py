import numpy as np 
import matplotlib.pyplot as plt
import mysql
import mysql.connector
class dlnet:
    def __init__(self, x, y):
        self.X=x #input
        self.Y=y #Desired Output
        print(self.Y.shape[1])
        self.Yh=np.zeros((1,self.Y.shape[1])) #Output of Network       
        self.L=2 #How much layer
        self.dims = [36, 37, 1]     #(input, hidden, output)   
        self.param = {}
        self.ch = {}
        self.grad = {}        
        self.loss = []
        self.lr=0.003
        self.sam = self.Y.shape[1]
    
    def nInit(self):    
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))                
        return  
    def forward(self):    
        Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
        A1 = self.relu(Z1)
        self.ch['Z1'],self.ch['A1']=Z1,A1
       
        Z2 = self.param['W2'].dot(A1) + self.param['b2']  
        A2 = self.Sigmoid(Z2)
        self.ch['Z2'],self.ch['A2']=Z2,A2        
        self.Yh=A2
        loss=self.nloss(A2)
        return self.Yh, loss
    def Sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    
    def relu(self,Z):
        return np.maximum(0,Z)
    
    def nloss(self,Yh):
        x= np.sum(Yh)
        a = 1-Yh
        a= np.sum(a)
        loss = (1./self.sam) * (-np.dot(self.Y,x) - np.dot(1-self.Y, a))    
        return loss
    def dRelu(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def dSigmoid(self,Z):
        s = 1/(1+np.exp(-Z))
        dZ = s * (1-s)
        return dZ
    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        print(dLoss_Yh.shape)
        dLoss_Z2 = dLoss_Yh * self.dSigmoid(self.ch['Z2'])    
        print(dLoss_Z2.shape)
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
       
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                            
        dLoss_Z1 = dLoss_A1 * self.dRelu(self.ch['Z1'])   
        print("dLossA1",self.dRelu(self.ch['Z1']).shape)
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        #print("X.shape", self.X.shape[1])
        #print("X.T shape", self.X.shape)
        self.X = np.reshape(np.array(self.X),(36,1))
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
        
        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
    def gd(self,X, Y):
        np.random.seed(1)                         
    
        self.nInit()
    
        
        Yh, loss=self.forward()
        self.backward()
        
        if i % 500 == 0:
            print ("Cost after iteration %i: %f" %(i, loss))
            self.loss.append(loss)
    
        return loss
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
        return np_del, trys,np_matrix.shape[0], np_matrix.shape[1]
    except:
        print ("Record failed to updated.")
        
def ScalingData(X):
    meanX = np.mean(X)
    stdDev = np.std(X)
    z= (X-meanX)/stdDev
    return z
inputs, outputs , row, column= AmbilDataMySQL()
inputs = ScalingData(inputs)
outputs = ScalingData(outputs)
shap = outputs.shape[0]
outputs = np.reshape(np.array(outputs),(shap,1))
iteration = 1000
lossarray = []
print("shape input:", inputs.shape, "\nshape output:", outputs.shape)
for i in range (0, iteration):
    for j in range(0, row):
        inputdata = inputs[j]
        outputdata = outputs[j]
        outputdata = np.reshape(np.array(outputdata),(1,1))
        nn = dlnet(inputdata,outputdata)
        loss = nn.gd(inputdata, outputdata)
        if i % 500 == 0:
            print ("Cost after iteration %i: %f" %(i, loss))
            lossarray.append(loss)