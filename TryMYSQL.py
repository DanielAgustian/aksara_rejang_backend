# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:20:05 2020

@author: Daniel
"""

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database = "dbskripsi"
)
label = "Exod"
std = 1445
mycursor = mydb.cursor()
print (mydb)
try:
    mycursor.execute("SELECT * FROM datakarakter")
    myresult = mycursor.fetchall()
    for x in myresult:
        print (x)
    #sql = "INSERT INTO datakarakter (label, karakter) VALUES (%s, %s)"
    #val = (label, std)
    #mycursor.execute(sql, val)   
    #mydb.commit()
    #print(mycursor.rowcount, "record inserted.")
except:
    print ("Failure")