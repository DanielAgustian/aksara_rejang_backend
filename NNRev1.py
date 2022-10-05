# -*- coding: utf-8 -*-
"""
@author: Daniel
"""
import mysql
import mysql.connector
import numpy as np
import math
from decimal import Decimal
import random
import sys
from copy import copy, deepcopy

maxEpoch = 1000
rasioPembelajaran = 0.5
momentum = 0.01
jumlahSarafInput = 37
jumlahSaratTersembunyi = 38
jumlahSaratOutput = 1

'''
    JSI adalah jumlah kriteria awal untuk NN yaitu 38 (hasil MPP dan label)
    JST harus lebih banyak daripada JSI, sehingga menggunakan 39
    JSO adalah jumlah kolom yang akan digunakan sebagai hasil.
'''
inputs = [0.00 for i in range(jumlahSarafInput)]
hOutput = [0.00 for i in range(jumlahSaratTersembunyi)]

outputs = [0.00 for i in range(jumlahSaratOutput)]
oGradient = [0.00 for i in range(jumlahSaratOutput)]
hGradient = [0.00 for i in range(jumlahSaratTersembunyi)]
ihDeltaBobotSebelumnya = [[0.00 for x in range(jumlahSaratTersembunyi)] for y in range(0, jumlahSarafInput)] 
hDeltaBiasSebelumnya = [0.00 for i in range(jumlahSaratTersembunyi)]
hoDeltaBobotSebelumnya = [[0.00 for x in range(jumlahSaratOutput)] for y in range(0, jumlahSaratTersembunyi)] 
oDeltaBiasSebelumnya = [0.00 for i in range(jumlahSaratOutput)]
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
            #print (matrix[indexRow])
            indexRow = indexRow+1            
        #print (matrix)
        '''    
        
        for r in matrix:
            print (r)
            '''
        return matrix
    except:
        print ("Record failed to updated.")
        
def NormalisasiData(matrix, baris, kolom): 
    rata2 = list(range(0, kolom))
    for kol in range(0, kolom):
        total = 0
        for bar in range (0,baris):
            total += matrix[bar][kol]
        rata2[kol] = total/baris
    stdDev = list(range(0, kolom))
    for kol in range(0, kolom):
        total = 0
        for bar in range(0, baris):
            total += (matrix[bar][kol]-rata2[kol])**2
        stdDev[kol] =  math.sqrt(total/baris)
    for kol in range(0,kolom):
        for bar in range(0, baris):
            matrix[bar][kol] =(matrix[bar][kol]-rata2[kol])/stdDev[kol]
            #======Normalisasi Data Selesai.=====================#
    return matrix
def InisialisasiBobot(jsi, jst, jso):
    jumlahBobot = (jsi*jst)+(jst*jso)+jst+jso
    print (jumlahBobot)     
    bobot = list(range(0, jumlahBobot))
    lo = 0.0001
    hi = 0.001
    for i in range(0, jumlahBobot):
        bobot[i] = (hi-lo)*random.random() + lo
        #print (bobot)
    return bobot, jumlahBobot

def setBobot(bobot, jumlahBobot):
    if len(bobot) != jumlahBobot:
        print("Data Tidak Sama")
        sys.exit()
        
    else:
        ihBobot = [[0 for x in range(jumlahSaratTersembunyi)] for y in range(jumlahSarafInput)]
        k = 0
        for i in range(0,jumlahSarafInput):
            for j in range(0, jumlahSaratTersembunyi):
                ihBobot[i][j] = bobot[k]
                k = k+1
        hBias = list(range(0, jumlahSaratTersembunyi))
        for i in range(0, jumlahSaratTersembunyi):
            hBias[i] = bobot[k]
            k= k+1
        hoBobot = [[0 for x in range(jumlahSaratOutput)] for y in range(jumlahSaratTersembunyi)]
        for i in range(0, jumlahSaratTersembunyi):
            for j in range(0, jumlahSaratOutput):
                hoBobot[i][j] = bobot[k]
                k = k+1
        oBias = list(range(0, jumlahSaratOutput))
        for i in range(0, jumlahSaratOutput):
            oBias[i] = bobot[k]
            k = k+1
        print("setBobot Done.") 
        return ihBobot, hBias, hoBobot, oBias
    
def BackPropagation(contohData, maxEpoch, rasio, momentum):
    contohDataKolomKriteria = list(range(0, jumlahSarafInput))
    contohDataKolomHasil = list(range(0, jumlahSaratOutput))
    urutanData = list(range(0, len(contohData)))
    panjangUrutan = len(urutanData)
    for i in range(0, panjangUrutan):
        urutanData[i] = i
    epoch  = 0
   
    while epoch < 1:
        epoch = epoch+1
        mse = MeanSquaredError(contohData)
       
        if mse<0.001:
            epoch = 1000
            break
        else:
            for i in range(0, panjangUrutan):
                ram = random.randrange(i, panjangUrutan)
                tmp = urutanData[ram]
                urutanData[ram] = urutanData[i]
                urutanData[i] = tmp
            
            for i in range(0, len(contohData)):
                idx = urutanData[i]
                for j in range(0, jumlahSarafInput-1):
                    contohDataKolomKriteria[j] = contohData[idx][j]
                for j in range(jumlahSarafInput-1, jumlahSaratOutput+jumlahSarafInput-1):
                    contohDataKolomHasil[j-jumlahSarafInput] = contohData[idx][j]
                hitungNilaiOutput(contohDataKolomKriteria)   
                if len(contohDataKolomHasil) != jumlahSaratOutput:
                    print("Panjang ContohDataKolomHasil Berbeda dengan Jumlah Saraf Output. ")
                else:
                    for j in range(0, len(oGradient)):
                        #turunanfungsiSoftmax = 0. KENAPA?
                        turunanfungsiSoftmax = (1-outputs[j])*outputs[j]
                        #print(turunanfungsiSoftmax)
                        oGradient[j] = turunanfungsiSoftmax * (contohDataKolomHasil[j]-outputs[j])
                    for j in range(0, len(hGradient)):
                        turunanFungsiHyperTan = (1 - hOutput[j]) * (1 + hOutput[j])
                        jumlahGradient = 0.000
                        for k in range(0, jumlahSaratOutput):
                            ex = oGradient[k]*hoBobot[j][k]
                            jumlahGradient += ex
                        hGradient[j] = turunanFungsiHyperTan*jumlahGradient
                    for j in range(0, len(ihBobot)):
                        for k in range(0, len(ihBobot[0])):
                            delta= rasioPembelajaran*hGradient[k]*inputs[j]
                            ihBobot[j][k] += delta
                            ihBobot[j][k] += momentum * ihDeltaBobotSebelumnya[j][k]
                            ihDeltaBobotSebelumnya[j][k] = delta
                    for j in range(0, len(hBias)):
                        delta =  rasioPembelajaran*hGradient[j]
                        hBias[j]+= delta
                        hBias[j] += momentum * hDeltaBiasSebelumnya[j]
                        hDeltaBiasSebelumnya[j] = delta
                    for j in range(0, len(hoBobot)):
                        for k in range(0,len(hoBobot[0])):
                            delta = rasioPembelajaran* oGradient[k] * hOutput[j]
                            hoBobot[j][k] += delta
                            hoBobot[j][k] += momentum * hoDeltaBobotSebelumnya[j][k]
                            hoDeltaBobotSebelumnya[j][k] = delta
                    for j in range(0, len(oBias)):
                        delta= rasioPembelajaran * oGradient[j] *1.0
                        oBias[j] += delta
                        oBias[j] += momentum * oDeltaBiasSebelumnya[j]
                        oDeltaBiasSebelumnya[j]= delta
        
       
           
def MeanSquaredError(contohData):
    
    contohDataKolomKriteria = list(range(0, jumlahSarafInput))
    contohDataKolomHasil = list(range(0,jumlahSaratOutput))
    hasil = 0.00
    for x in range(0, len(contohData)):
        for i in range(0, jumlahSarafInput-1):
            contohDataKolomKriteria[i] = contohData[x][i]
        for i in range(jumlahSarafInput-1, jumlahSaratOutput+jumlahSarafInput-1):
            contohDataKolomHasil[i-jumlahSarafInput] = contohData[x][i]
        dataKolomHasil = hitungNilaiOutput(contohDataKolomKriteria)
        for j in range(0, len(dataKolomHasil)):
            hasil+=(dataKolomHasil[j]-contohDataKolomHasil[j])**2
            
    hasil = hasil/len(contohData)
    print ("hasil MSE: ", hasil)
    return hasil

def hitungNilaiOutput(mInput):
        if len(mInput) != jumlahSarafInput:
            print("pada fungsi hitungNilaiOutput, panjang array tidak sama")
            sys.exit()
        else: 
            hjumlahBobotdanBias = list(range(0, jumlahSaratTersembunyi))
            for i in range(0, jumlahSaratTersembunyi):
                hjumlahBobotdanBias[i] = 0.00
            ojumlahBobotdanBias = list(range(0, jumlahSaratOutput))
            for i in range(0, jumlahSaratOutput):
                ojumlahBobotdanBias[i] = 0.00
                
            for i in range(0, len(mInput)):
                inputs[i] = mInput[i]             
            for j in range(0, jumlahSaratTersembunyi):
                for i in range(0, jumlahSarafInput):
                    hjumlahBobotdanBias[j] += inputs[i]*ihBobot[i][j]        
           
            for i in range(0, jumlahSaratTersembunyi):
                hjumlahBobotdanBias[i] += hBias[i]
            
            for i in range(0, jumlahSaratTersembunyi):
                hOutput[i] = HyperTan(hjumlahBobotdanBias[i])
            for j in range(0, jumlahSaratOutput):
                for i in range(0, jumlahSaratTersembunyi):
                    ojumlahBobotdanBias[j] += hOutput[i] *hoBobot[i][j]
            for i in range(0, jumlahSaratOutput):
                ojumlahBobotdanBias[i] += oBias[i]
            softOut = Softmax(ojumlahBobotdanBias)
            #print ("oJumlahBobotdanBias = ", ojumlahBobotdanBias)
            for i in range(0, len(softOut)):
                outputs[i] = softOut[i]
            
            return outputs    
            
def HyperTan(x):
    
    if x<-20.00:
        return -1.00
    elif x>20.0 :
        return 1.00
    else:
        tanh = np.tanh(x)
        return tanh
    
def Softmax(oJumlahBobotdanBias): 
    panjang = len(oJumlahBobotdanBias)
    maks = oJumlahBobotdanBias[0]
    for i in range(0, panjang):
        if oJumlahBobotdanBias[i]>maks:
            maks= oJumlahBobotdanBias[i]
    
    skala = 0.00
    for i in range(0, panjang):
        skala += math.exp(oJumlahBobotdanBias[i]-maks)  
    print ("Softmax DBG", skala)
    hasil = [0.00 for i in range( panjang)]
    for i in range(0, panjang):
        hasil[i] = math.exp(oJumlahBobotdanBias[i]-maks)/skala
    
     
    return hasil
    
#================    MAIN ACTIVITY   ====================================#    
matrix = AmbilDataMySQL()
globalRows = len(matrix)
globalCols = len(matrix[0])
matrixNormalisasi = NormalisasiData(matrix, globalRows, globalCols)
bobot, jumlahBobot = InisialisasiBobot(jumlahSarafInput, jumlahSaratTersembunyi, jumlahSaratOutput)
ihBobot, hBias, hoBobot, oBias = setBobot(bobot,jumlahBobot)


BackPropagation(matrixNormalisasi, maxEpoch, rasioPembelajaran, momentum)
print (outputs)