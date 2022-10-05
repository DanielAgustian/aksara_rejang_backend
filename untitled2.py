# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:14:48 2020

@author: Daniel
"""
import mysql.connector
import cv2
import numpy as np
import MySQLdb
def beginningmysql(label, matrix):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database = "dbskripsi"
    )
    mycursor = mydb.cursor()
    try:
        sql2 ="Select * from namakarakter"
        mycursor.execute(sql2)
        records = mycursor.fetchall()
        for row in records:
            if label == row[1]:
                print ("Kode untuk ",label,"adalah",row[0])
                label = row[0]
    except(MySQLdb.Error, MySQLdb.Warning) as e:
        print(e)
    try:
        state = "label,nilaikebenaran,k0,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24,k25,k26,k27,k28,k29,k30,k31,k32,k33,k34,k35"
        
        sql = "INSERT INTO datakarakter (label,k0,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24,k25,k26,k27,k28,k29,k30,k31,k32,k33,k34,k35) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        val = (label, matrix[0],matrix[1],matrix[2],matrix[3],matrix[4],matrix[5],matrix[6],matrix[7],matrix[8],matrix[9],matrix[10],matrix[11],
               matrix[12],matrix[13],matrix[14],matrix[15],matrix[16],matrix[17],matrix[18],matrix[19],matrix[20],matrix[21],matrix[22],matrix[23]
               ,matrix[24],matrix[25],matrix[26],matrix[27],matrix[28],matrix[29],matrix[30],matrix[31],matrix[32],matrix[33],matrix[34],matrix[35])
        print(sql)
        
        mycursor.execute(sql, val)   
        mydb.commit()
        print(mycursor.rowcount, "record inserted.")
    except(MySQLdb.Error, MySQLdb.Warning) as e:
        print (e)
        print ("Record failed to updated.")
        
     
        
      
#Text = "image.jpg"



print ("karakter pada Gambar adalah ...")
#label = input()
label = "ya"

print ("Nilai Kebenaran dari Gambar? Yes=1 No=0")
#kebenaran = input()

Text = "karakter/ya/ya10.jpg" 
#Text = "image.jpg" 
img = cv2.imread(Text)
print("LINK : "+Text+"\n")
## (1) Convert to gray, and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

## (2) Morph-op to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

## (3) Find the max-area contour
cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnt = sorted(cnts, key=cv2.contourArea)[-1]

## (4) Crop and save it
x,y,w,h = cv2.boundingRect(cnt)
dst = img[y:y+h, x:x+w]

th, threshed = cv2.threshold(dst, 240, 255, cv2.THRESH_BINARY)
cv2.imwrite("resultant.png", dst)

##=================Pixel Population matrix Begins======================##

ordo = 6
bagianRow = list(range(0, ordo+1))
bagianColumn = list(range(0, ordo+1))
sizeofArray = threshed.shape[0] 
jumlahRow = sizeofArray/ordo
sizeofArray = threshed.shape[1]
jumlahColumn = sizeofArray/ordo
for i in range (0,ordo+1):
    if i==0:
        bagianRow[i] = 0
        bagianColumn[i] = 0
    else:
        bagianRow[i] = round((jumlahRow*i))
        bagianColumn[i] = round((jumlahColumn*i))
print (bagianRow)
print (bagianColumn)
print (dst.shape)
index = 0
StringToDB = ""
dataKarakter = list(range(0, 36))
for j in range(0, ordo):
    for i in range (0, ordo):
        allBlack = 0
        allPixel = 0
        
        for y in range(bagianColumn[j], bagianColumn[j+1]):
            for x in range(bagianRow[i], bagianRow[i+1]):
                if threshed[x,y,0] == 0:
                    allBlack = allBlack+1
                allPixel = allPixel+1      
        popPercentage = round(allBlack/allPixel, 3)
        dataKarakter[index] = popPercentage
        popPercentage = str(popPercentage)
        index = index+1
        #print ("Jumlah Piks Hitam",index,": ", allBlack)
        #print ("Jumlah Piksel",index,":", allPixel)
        #print ("Persenan populasi piksel: ", popPercentage)
        #print ("Integer yang dimasukan ke DB: ", StringToDB)
        #print ("STRING TO DB", StringToDB)
        #print ("===========================================================")

beginningmysql(label, dataKarakter)
#Splitting = StringToDB.split()
#print (Splitting[0])
