# -- coding: utf-8 --
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

"""VERİLER YÜKLENİYOR VE DÜZENLENİYOR..."""
veriler = pd.read_excel('veri_duzenlenmis.xlsx')
       
sag_sol = veriler[[1]]
le = preprocessing.LabelEncoder()
sag_sol = le.fit_transform(sag_sol)
# print(sag_sol)
kalan= veriler.iloc[:,2:].values
# print(kalan)
bir= pd.DataFrame(data= sag_sol, index= range(271), columns= ["el"])
iki= pd.DataFrame(data= kalan, index= range(271))
veri=pd.concat([bir,iki], axis=1)
# print(veri)


"""EĞİTİM/TEST VERİLERİ AYRILIYOR..."""
x_train, x_test,y_train,y_test = train_test_split(kalan,sag_sol,test_size=0.33, random_state=0)

"""Ölçekleme GNB için sonuçları olumsuz etkiledi."""
# sc=StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)


"""ÖZNİTELİK SEÇİMİ..."""
from sklearn.feature_selection import SelectKBest, chi2
select= SelectKBest(chi2, k=1500).fit(x_train, y_train)
x_train = select.transform(x_train)
x_test = select.transform(x_test)

    
print("********** ALGORİTMALAR DENENİYOR ***************")  

#LOGİSTİC REGRESSİON
from sklearn.linear_model import LogisticRegression
print('LGR')
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred = logr.predict(x_test)
acc= accuracy_score(y_test, y_pred)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nLGR",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#KNN
from sklearn.neighbors import KNeighborsClassifier
print('KNN')
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
acc= accuracy_score(y_test, y_pred)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nKNN",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#SVC
from sklearn.svm import SVC
print('SVC')
svc = SVC(kernel='rbf')
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
acc= accuracy_score(y_test, y_pred)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nSVC",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#NAİVEBAYES
from sklearn.naive_bayes import GaussianNB
print('GNB')
gnb = GaussianNB()  
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
acc= accuracy_score(y_pred,y_test)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nGNB",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#KARARAGACI
from sklearn.tree import DecisionTreeClassifier
print('DTC')
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(x_train,y_train)

y_pred = dtc.predict(x_test)
acc= accuracy_score(y_test, y_pred)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nDTC",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("*******")
#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
print('RFC')
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state=500)
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
acc= accuracy_score(y_pred,y_test)
print("DOĞRULUK: ", acc)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print("\nRFC",classification_report(y_test, y_pred))
print("Roc eğrisi altındaki alan (AUC): ", roc_auc_score(y_test, y_pred))


print("****** CROSS VALUDATION *********")
#CROSS VALIDATION
from sklearn.model_selection import cross_val_score
k=10

crossval =cross_val_score(logr, X= x_train, y= y_train, cv=k)
print("LOGR Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(svc, X= x_train, y= y_train, cv=k)
print("SVC Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(rfc, X= x_train, y= y_train, cv=k)
print("RFC Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(dtc, X= x_train, y= y_train, cv=k)
print("DTC Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(gnb, X= x_train, y= y_train, cv=k)
print("GNB Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))


crossval =cross_val_score(knn, X= x_train, y= y_train, cv=k)
print("KNN Cross Validation: ", np.mean(crossval))
print("Cross Validation Standart Sapma: ",np.std(crossval))
