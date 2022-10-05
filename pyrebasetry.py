# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 21:41:17 2020

@author: Daniel
"""
import pyrebase
import numpy as np 
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
W1 = np.random.randn(2, 2)
W1 = W1.tolist()
data = [["Parwiz Forough", "Ameria"],["Hathor", "Megane","Aluminum"]]
'''
try:    
    db.child("users").push(W1)#push data to generated new key
    #db.child("users").child("OwnKey").set(data)# push data to ourself key.
    #db.child("users").child("OwnKey").update(data) #to update data
    
    print("Data Pushed.")
    #users = db.child("users").get()
    #print(users.val())
    
except:
    print("Error")
    '''