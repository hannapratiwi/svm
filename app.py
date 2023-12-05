import numpy as np
import pandas as pd
import streamlit as st
import sklearn
import pickle
import re
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report

#model
#temp_model = loaded_model = pickle.load(open('model_svm.sav', 'rb'))

#preprocessing data
clean = pd.read_csv("preprocessingfixbanget.csv")

st.header ("GSite")

#tema = ["Portofolio", "Marketing", "Company Profile"]
#tema_selected = st.selectbox("Silahkan pilih tema", options = tema)
#tema_mapping = {
#     "Portofolio": 1,
#     "Marketing": 2,
#     "Company Profile": 3
# }
#selected_value = tema_mapping.get(tema_selected)
tema = st.text_input("Tentukan tema")
page = st.text_input("Jumlah halaman")
menu = st.text_input("Jumlah menu")
font = st.text_input("Tentukan font, contoh Open Sans")
warna = st.text_input("Masukan Warna")

input = [tema, page, menu, font, warna]
i_font = input[3]
i_warna = input[4]

#data train
train = clean[["tema","page", "menu", "font", "warna"]]

#label train
label = clean[["label"]]

#data testing
test = pd.read_csv("testbaru.csv")

#data testing
Xtest = test[["tema","page", "menu", "font", "warna"]]

#label testing
Ytest = test[["label"]]

#Analisa warna
#data training
tag = []
for warna in train["warna"]:
    tmp = re.sub('[^A-Za-z0-9#]+', ' ', warna)
    tmp = tmp.split(" ")
    tmp = list(filter(None, tmp))
    tag.extend(tmp)
warna = np.array(tag)
warna_values, warna_counts = np.unique(warna, return_counts=True)
tf_warna = pd.DataFrame(data = warna_counts, index = warna_values)
tf_warna.sort_values(by=tf_warna.columns[0], ascending=False)
#tf_warna.to_csv('warna.csv')

#Analisa font
#data training
code = []
for family in train["font"]:
    tmp = re.sub('[^A-Za-z0-9]+', ' ', family.lower())
    tmp = tmp.split(" ")
    tmp = list(filter(None, tmp))
    code.extend(tmp)
font = np.array(code)
font_values, font_counts = np.unique(font, return_counts=True)
tf_font = pd.DataFrame(data = font_counts, index = font_values)
# tf_font.to_csv('font.csv')
tf_font.sort_values(by=tf_font.columns[0], ascending=False)

#one hot encoding css
#warna
#data training
tag_warna = pd.read_csv("warna.csv")
clean_warna = list(tag_warna.warna.to_numpy())

#data training
ohe = []
for warna in train["warna"]:
    tmp = [0] * len(clean_warna)
    for idy, y in enumerate(clean_warna):
        y = y.split(" ")
        for x in y:
            if(x in warna):
                tmp[idy] = 1
    ohe.append(tmp)
ohe_warna = pd.DataFrame(data = ohe,  columns = clean_warna)

#data testing
ohe = []
for warna in Xtest["warna"]:
    tmp = [0] * len(clean_warna)
    for idy, y in enumerate(clean_warna):
        y = y.split(" ")
        for x in y:
            if(x in warna):
                tmp[idy] = 1
    ohe.append(tmp)
test_warna = pd.DataFrame(data = ohe,  columns = clean_warna)

# input warna
tmp = []
for idy, y in enumerate(clean_warna):
  ly = y.split(" ")
  cek = False
  for x in ly:
    if(x in i_warna):
      cek =True
  if cek:
    tmp.append(1)
  else:
    tmp.append(0)
ci_warna = tmp

#font
#data taining
tag_font = pd.read_csv("font.csv")
clean_font = list(tag_font.font.to_numpy())

#data training
ohe = []
for family in train["font"]:
    tmp = [0] * len(clean_font)
    for idy, y in enumerate(clean_font):
        y = y.split(" ")
        for x in y:
            if(x in family):
                tmp[idy] = 1
    ohe.append(tmp)
ohe_font = pd.DataFrame(data = ohe, columns = clean_font)

#data testing
ohe = []
for family in Xtest["font"]:
    tmp = [0] * len(clean_font)
    for idy, y in enumerate(clean_font):
        y = y.split(" ")
        for x in y:
            if(x in family):
                tmp[idy] = 1
    ohe.append(tmp)
test_font = pd.DataFrame(data = ohe, columns = clean_font)

# input font
tmp = []
for idy, y in enumerate(clean_font):
  ly = y.split(" ")
  for x in ly:
    cek = False
    if(x in i_font):
      cek = True
  if cek:
    tmp.append(1)
  else:
    tmp.append(0)
ci_font = tmp

#spliting data
# data training
dclean = clean[["tema","page", "menu"]]
dclean = dclean.join(ohe_font)
dclean = dclean.join(ohe_warna)

# data testing
dtest = Xtest[["tema", "page", "menu"]]
dtest = dtest.join(test_font)
dtest = dtest.join(test_warna)
test = dtest.values.tolist()

dinput = [input[0], input[1], input[2]]
dinput.extend(ci_font)
dinput.extend(ci_warna)

#Model SVM
#memasukan data training ke svm
model = svm.SVC()
model.fit(dclean, label.values.ravel())

# # input prediksi
# #code predict
generate = ''

if st.button ("generate") :
    generate = model.predict([dinput])
