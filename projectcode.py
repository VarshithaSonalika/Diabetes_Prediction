# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:28:28 2021

@author: Varshitha Sonalika
"""

#import all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve

###LOADING THE DATASET
df = pd.read_csv(r"C:\Users\Varshitha Sonalika\Downloads\diabetes.csv")
print("\ndataset: \n",df)
x1 = df.head() #displays first 5 records of the data
print("\nfirst 5 records of the dataset: \n",x1)
x2 = df.tail() #displays last 5 records of the data
print("\nlast 5 records of the dataset: \n",x2)
x3 = df.sample() #displays randomly any record of the data
print("\nRandomly displays any record of the data: \n",x3)
x4 = df.shape #shape of the data
print("\nshape of the data set: \n",x4)
x5 = df.dtypes #lists types of all variables
print("\ntypes of all variables: \n",x5)
x6 = df.info() #finding out whether dataset contains any null values or not
print("\nNull values in the dataset: ",x6)
x7 = df.describe() #statistical summary of the dataset
print("\nstatistical summary of the data: \n",x7)

###DATA CLEANING
print("\n DATA CLEANING \n")
df = df.drop_duplicates() #dropping duplicates
x8 = df.shape
if x8 == x4:
    print("\nThere are no duplicates in the dataset. \n")
else:
    print("\nsome duplicates are present in the dataset. \n")
x9 = df.isnull().sum() #checking no.of null values
print("\n no.of null values in each column: \n",x9)

###CHECKING NO.OF ZERO VALUES IN DATASET
print(" \n CHECKING NO.OF ZERO VALUES IN DATASET \n")
print('\nNo.of zero values in Pregnancies : ', df[df['Pregnancies']==0].shape[0])
print('No.of zero values in Glucose : ', df[df['Glucose']==0].shape[0])
print('No.of zero values in BloodPressure : ', df[df['BloodPressure']==0].shape[0])
print('No.of zero values in SkinThickness : ', df[df['SkinThickness']==0].shape[0])
print('No.of zero values in Insulin : ', df[df['Insulin']==0].shape[0])
print('No.of zero values in BMI : ', df[df['BMI']==0].shape[0])
print('No.of zero values in DiabetesPedigreeFunction : ', df[df['DiabetesPedigreeFunction']==0].shape[0])
print('No.of zero values in Age : ', df[df['Age']==0].shape[0])

###ANALYSING THE OUTCOME COLUMN
print("ANALYSING THE OUTCOME COLUMN \n")
(Negative,Positive) = df['Outcome'].value_counts()
print("\nNo.of samples tested negative(0) for diabetes is : ",Negative)
print("No.of samples tested positive(1) for diabetes is : ", Positive)
my_labels='Negative(0)','Positive(1)'
print('Outcome pieplot')
plt.pie((Negative,Positive),labels=my_labels,autopct='%1.1f%%')
plt.title('Outcome plot')
plt.show()

###DATA VISUALIZATION
print("\n DATA VISUALIZATION \n")
print('BOX-PLOT OF DATA')
df.boxplot(figsize = (13,5))
plt.title("box-plot of the data")
plt.show()
print('SCATTER-PLOT OF DATA')
scatter_matrix(df,figsize=(13,13))
plt.title("scatter-plot of the data")
plt.show()
print('HISTOGRAM OF DATA')
df.hist(bins=10, figsize=(10,10))
plt.title("histogram of the data")
plt.show()
print('PAIR-PLOT OF DATA')
sns.pairplot(data=df,hue='Outcome')
plt.title("pairplot of the data")
plt.show()

###REPLACING NO.OF ZERO VALUES WITH MEAN OF THE COLUMNS
X = df.drop('Outcome', axis=1)
Y = df['Outcome']
X.replace(0,value=df.mean(),inplace=True)
x = X.describe()
print("\n statistical summary of the data after replacing no.of zero values : \n",x)


###CHECKING ZERO VALUES IN THE DATASET 
print('No.of zero values in Pregnancies : ', X[X['Pregnancies']==0].shape[0])
print('No.of zero values in Glucose : ', X[X['Glucose']==0].shape[0])
print('No.of zero values in BloodPressure : ', X[X['BloodPressure']==0].shape[0])
print('No.of zero values in SkinThickness : ', X[X['SkinThickness']==0].shape[0])
print('No.of zero values in Insulin : ', X[X['Insulin']==0].shape[0])
print('No.of zero values in BMI : ', X[X['BMI']==0].shape[0])
print('No.of zero values in DiabetesPedigreeFunction : ', X[X['DiabetesPedigreeFunction']==0].shape[0])
print('No.of zero values in Age : ', X[X['Age']==0].shape[0])
X.boxplot(figsize=(13,5))
plt.title("boxplot for checking outliers")
plt.show()

###IDENTIFING AND REMOVING THE OUTLIERS
print("\n IDENTIFING OUTLIERS \n")
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75) 
IQR = Q3-Q1
#print("Inter quartile range : \n", IQR)

median_p = int(X["Pregnancies"].median())
X["Pregnancies"] = np.where(X["Pregnancies"]>6, median_p, X['Pregnancies'])

median_g = int(X["Glucose"].median())
X["Glucose"] = np.where(X["Glucose"]<50, median_g, X['Glucose'])

median_bp = int(X["BloodPressure"].median())
X["BloodPressure"] = np.where(X["BloodPressure"]<50, median_bp, X['BloodPressure'])

median_s = int(X["SkinThickness"].median())
X["SkinThickness"] = np.where(X["SkinThickness"]>median_s, median_s, X['SkinThickness'])

median_i = int(X["Insulin"].median())
X["Insulin"] = np.where(X["Insulin"]>250, median_i, X['Insulin'])
median_in = int(X["Insulin"].median())
X["Insulin"] = np.where(X["Insulin"]<40, median_in, X['Insulin'])


median_bmi = float(X["BMI"].median())
X["BMI"] = np.where(X["BMI"]>median_bmi, median_bmi, X['BMI'])

median_dpf = float(X["DiabetesPedigreeFunction"].median())
X["DiabetesPedigreeFunction"] = np.where(X["DiabetesPedigreeFunction"]>median_dpf, median_dpf, X['DiabetesPedigreeFunction'])

median_age = int(X["Age"].median())
X["Age"] = np.where(X["Age"]>median_age, median_age, X['Age'])

print("shape of the data after removing outliers : ", X.shape)
print("statistical summary of the data after removing outliers : \n",X.describe())
X.boxplot(figsize=(13,5))
plt.title("boxplot after removing outliers")
plt.show()

###APPLYING FEATURE SCALER
std = StandardScaler() #Apply standard scaler
std.fit(X)
X_std = std.fit_transform(X)

###SPLITTING INTO TRAIN AND TEST DATA
print("\n TRAIN AND TEST DATA\n")
X_train, X_test, Y_train, Y_test = train_test_split(X_std,Y,test_size=0.25,random_state=2)
#Finding the shapes of test and train data
print("X_train : ", X_train.shape)
print("Y_train : ", Y_train.shape)
print("X_test : ", X_test.shape)
print("Y_test : ", Y_test.shape)

###TRAINING OUR MODEL
##LOGISTIC REGRESSION
print("\n LOGISTIC REGRESSION \n")
LR = LogisticRegression(solver='liblinear', multi_class='ovr')
LR.fit(X_train, Y_train)
print(LR.fit(X_train, Y_train))
LR_Ypred = LR.predict(X_test)
print("lr_ypred : \n", LR_Ypred)
print("y_test :  \n",Y_test)
print("Train data accuracy of logistic regression : ", LR.score(X_train,Y_train)*100)
print("Test data accuracy of logistic regression : ", LR.score(X_test,Y_test)*100)

##KNN(KNeighbors Classifier)
print("\n KNEIGHBORS CLASSIFIER \n")
KNN = KNeighborsClassifier()
print(KNN.fit(X_train, Y_train))
KNN_Ypred = KNN.predict(X_test)
print("knn_ypred : \n",KNN_Ypred)
print("y_test : \n",Y_test)
print("Train data accuracy of KNN(KNeighbors Classifier) : ", KNN.score(X_train,Y_train)*100)
print("Test data accuracy of KNN(KNeighbors Classifier) : ", KNN.score(X_test,Y_test)*100)

##NAVIE-BAYES CLASSIFIER
print("\n NAVIE-BAYES CLASSIFIER \n")
NB = GaussianNB()
print(NB.fit(X_train, Y_train))
NB_Ypred = NB.predict(X_test)
print("nb_ypred : \n",NB_Ypred)
print("y_test : \n",Y_test)
print("Train data accuracy of navie-bayes classifier : ", NB.score(X_train,Y_train)*100)
print("Test data accuracy of navie-bayes classifier : ", NB.score(X_test,Y_test)*100)

##SVM(Support Vector Machine)
print("\n SUPPORT VECTOR MACHINE(SVM) \n")
SV = SVC()
print(SV.fit(X_train, Y_train))
SV_Ypred = SV.predict(X_test)
print("sv_ypred : \n",SV_Ypred)
print("y_test : \n",Y_test)
print("Train data accuracy of SVM(Support Vector Machine) : ", SV.score(X_train,Y_train)*100)
print("Test data accuracy of  SVM(Support Vector Machine) : ", SV.score(X_test,Y_test)*100)

##DECISION TREE
print("\n DECISION TREE CLASSIFIER \n")
DT = DecisionTreeClassifier()
print(DT.fit(X_train, Y_train))
DT_Ypred = DT.predict(X_test)
print("dt_ypred : \n",DT_Ypred)
print("y_test : \n",Y_test)
print("Train data accuracy of decision tree : ", DT.score(X_train,Y_train)*100)
print("Test data accuracy of decision tree : ", DT.score(X_test,Y_test)*100)

##RANDOM FOREST
print("\n RANDOM FOREST CLASSIFIER \n")
RF = RandomForestClassifier()
print(RF.fit(X_train, Y_train))
RF_Ypred = RF.predict(X_test)
print("rf_ypred : \n",RF_Ypred)
print("y_pred : \n",Y_test)
print("Train data accuracy of random forest : ", RF.score(X_train,Y_train)*100)
print("Test data accuracy of random forest : ", RF.score(X_test,Y_test)*100)


########################################################################
########################################################################
########################################################################

###CONFUSION MATRIX OF "LOGISTIC REGRESSION"
print("\n LOGISTIC REGRESSION ANALYSIS \n")
cmLR = confusion_matrix(Y_test, LR_Ypred)
print("\nConfusion matrix of Logistic regression is : \n",cmLR)
print("\nclassification report of logistic regression: \n", classification_report(Y_test,LR_Ypred,digits=4))
TN_LR = cmLR[0,0]
print("True negative of logistic regression confusion matrix : ",TN_LR)
FP_LR = cmLR[0,1]
print("False positive of logistic regression confusion matrix : ",FP_LR)
FN_LR = cmLR[1,0]
print("False negative of logistic regression confusion matrix : ",FN_LR)
TP_LR = cmLR[1,1] 
print("True positive of logistic regression confusion matrix : ",TP_LR)
print("Accuracy rate of logistic regression {} ".format(np.divide(np.sum([cmLR[0,0],cmLR[1,1]]),np.sum(cmLR))*100))
print("Misclassification rate of logistic regression:{}".format(np.divide(np.sum([cmLR[0,1],cmLR[1,0]]),np.sum(cmLR))*100))
table_LR = pd.crosstab(Y_test, LR_Ypred, rownames=['Actual Values'],colnames=['predicted values'], margins=True)
print("\n Logistic regression cross table :")
print(table_LR)
print("\n Accuracy score of logistics regression : ", accuracy_score(Y_test, LR_Ypred)*100)


###PRECISION(PPV-Positive predicted value)
precision_LR = TP_LR/(TP_LR+FP_LR)
precision_score_LR = TP_LR/float(TP_LR+FP_LR)*100
print('Precision score of logistic regression:{0:0.4f}'.format(precision_score_LR))
###RECALL(TPR-True positive rate)
recall_score_LR = TP_LR/(TP_LR+FN_LR)
recall_score_LR = TP_LR/float(TP_LR+FN_LR)*100
print('recall or sensitivity score of logistics regression:{0:0.4f}'.format(recall_score_LR))
###FALSE POSITIVE RATE(FPR)
fpr_LR = FP_LR/(FP_LR+TN_LR)
fpr_LR = FP_LR/float(FP_LR+TN_LR)*100
print('false positive rate of logistic regression:{0:0.4f}'.format(fpr_LR))
###SPECIFICITY
specificity_LR = TN_LR/(TN_LR+FP_LR)
specificity_LR = TN_LR/float(TN_LR+FP_LR)*100
print('specificity of logistic regression:{0:0.4}'.format(specificity_LR))
###F1_SCORE
print("f1_score of logistic regression is : ",f1_score(Y_test, LR_Ypred)*100)

###ROC AND ROC AUC
#To check the performance of a clssification model
auc_LR = roc_auc_score(Y_test, LR_Ypred)
print("ROC_AUC_SCORE of Logistic regression : ", auc_LR)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(Y_test, LR_Ypred)
plt.plot(fpr_LR, tpr_LR, color='orange', label='ROC of logistics regression')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve of logistic regression(area=%0.2f')
plt.xlabel('False Positive Rate of logistic regression')
plt.ylabel('True positive Rate of logistic regression')
plt.title("Receiver Operating Characteristic(ROC) curve of logistic regression")
plt.legend()
plt.grid()
plt.show()

####################################################
#######################################################
################################################33

###CONFUSION MATRIX OF "KNN"
print("\n KNEIGHBORS CLASSIFICATION ANALYSIS \n")
cmKNN = confusion_matrix(Y_test, KNN_Ypred)
print("\nConfusion matrix of  KNN(KNeighbors Classifier) is \n",cmKNN)
print("\nclassification report of KNN(KNeighbors Classifier) : \n", classification_report(Y_test,KNN_Ypred,digits=4))
TN_KNN = cmKNN[0,0]
print("True negative of KNN(KNeighbors Classifier) confusion matrix : ",TN_KNN)
FP_KNN = cmKNN[0,1]
print("False positive of KNN(KNeighbors Classifier) confusion matrix : ",FP_KNN)
FN_KNN = cmKNN[1,0]
print("False negative of KNN(KNeighbors Classifier) confusion matrix : ",FN_KNN)
TP_KNN = cmKNN[1,1] 
print("True positive of KNN(KNeighbors Classifier) confusion matrix : ",TP_KNN)
print("Accuracy rate of KNN(KNeighbors Classifier) {} ".format(np.divide(np.sum([cmKNN[0,0],cmKNN[1,1]]),np.sum(cmKNN))*100))
print("Misclassification rate:{}".format(np.divide(np.sum([cmKNN[0,1],cmKNN[1,0]]),np.sum(cmKNN))*100))
table_KNN = pd.crosstab(Y_test, KNN_Ypred, rownames=['Actual Values'],colnames=['predicted values'], margins=True)
print("\n KNN(KNeighbors Classifier) cross table :")
print(table_KNN)
print("\n Accuracy score of KNN(KNeighbors Classifier) : ", accuracy_score(Y_test, KNN_Ypred)*100)


###PRECISION(PPV-Positive predicted value)
precision_KNN = TP_KNN/(TP_KNN+FP_KNN)
precision_score_KNN = TP_KNN/float(TP_KNN+FP_KNN)*100
print('Precision score of KNN(KNeighbors Classifier):{0:0.4f}'.format(precision_score_KNN))
###RECALL(TPR-True positive rate)
recall_score_KNN = TP_KNN/(TP_KNN+FN_KNN)
recall_score_KNN = TP_KNN/float(TP_KNN+FN_KNN)*100
print('recall or sensitivity score of KNN(KNeighbors Classifier):{0:0.4f}'.format(recall_score_KNN))
###FALSE POSITIVE RATE(FPR)
fpr_KNN = FP_KNN/(FP_KNN+TN_KNN)
fpr_KNN = FP_KNN/float(FP_KNN+TN_KNN)*100
print('false positive rate of KNN(KNeighbors Classifier):{0:0.4f}'.format(fpr_KNN))
###SPECIFICITY
specificity_KNN = TN_KNN/(TN_KNN+FP_KNN)
specificity_KNN = TN_KNN/float(TN_KNN+FP_KNN)*100
print('specificity of KNN(KNeighbors Classifier):{0:0.4}'.format(specificity_KNN))
###F1_SCORE
print("f1_score of KNN(KNeighbors Classifier) is : ",f1_score(Y_test, KNN_Ypred)*100)

###ROC AND ROC AUC
#To check the performance of a clssification model
auc_KNN = roc_auc_score(Y_test, KNN_Ypred)
print("ROC_AUC_SCORE of KNN(KNeighbors Classifier) : ", auc_KNN)
fpr_KNN, tpr_KNN, thresholds_KNN = roc_curve(Y_test, KNN_Ypred)
plt.plot(fpr_KNN, tpr_KNN, color='orange', label='ROC of KNN(KNeighbors Classifier)')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve of KNN(KNeighbors Classifier)(area=%0.2f')
plt.xlabel('False Positive Rate of KNN(KNeighbors Classifier)')
plt.ylabel('True positive Rate of KNN(KNeighbors Classifier)')
plt.title("Receiver Operating Characteristic(ROC) curve of KNN(KNeighbors Classifier)")
plt.legend()
plt.grid()
plt.show()

#######################################################
#######################################################
######################################################

###CONFUSION MATRIX OF "Navie-Bayes"
print("\n NAVIE-BAYES CLASSIFICATION ANALYSIS \n")
cmNB = confusion_matrix(Y_test, NB_Ypred)
print("\nConfusion matrix of navie-bayes classifier is \n",cmNB)
print("classification report of navie-bayes classifier : \n", classification_report(Y_test,NB_Ypred,digits=4))
TN_NB = cmNB[0,0]
print("True negative of navie-bayes classifier confusion matrix : ",TN_NB)
FP_NB = cmNB[0,1]
print("False positive of navie-bayes classifier confusion matrix : ",FP_NB)
FN_NB = cmNB[1,0]
print("False negative of navie-bayes classifier confusion matrix : ",FN_NB)
TP_NB = cmNB[1,1] 
print("True positive of navie-bayes classifier confusion matrix : ",TP_NB)
print("Accuracy rate of navie-bayes classifier {} ".format(np.divide(np.sum([cmNB[0,0],cmNB[1,1]]),np.sum(cmNB))*100))
print("Misclassification rate of navie-bayes classifier:{}".format(np.divide(np.sum([cmNB[0,1],cmNB[1,0]]),np.sum(cmNB))*100))
table_NB = pd.crosstab(Y_test, NB_Ypred, rownames=['Actual Values'],colnames=['predicted values'], margins=True)
print("\n navie-bayes classifier cross table : ")
print(table_NB)
print("\n Accuracy score of navie-bayes classifier : ", accuracy_score(Y_test, NB_Ypred)*100)


###PRECISION(PPV-Positive predicted value)
precision_NB = TP_NB/(TP_NB+FP_NB)
precision_score_NB = TP_NB/float(TP_NB+FP_NB)*100
print('Precision score of navie-bayes classifier:{0:0.4f}'.format(precision_score_NB))
###RECALL(TPR-True positive rate)
recall_score_NB = TP_NB/(TP_NB+FN_NB)
recall_score_NB = TP_NB/float(TP_NB+FN_NB)*100
print('recall or sensitivity score of navie-bayes classifier:{0:0.4f}'.format(recall_score_NB))
###FALSE POSITIVE RATE(FPR)
fpr_NB = FP_NB/(FP_NB+TN_NB)
fpr_NB= FP_NB/float(FP_NB+TN_NB)*100
print('false positive rate of navie-bayes classifier:{0:0.4f}'.format(fpr_NB))
###SPECIFICITY
specificity_NB = TN_NB/(TN_NB+FP_NB)
specificity_NB = TN_NB/float(TN_NB+FP_NB)*100
print('specificity of navie-bayes classifier:{0:0.4}'.format(specificity_NB))
###F1_SCORE
print("f1_score of navie-bayes classifier is : ",f1_score(Y_test, NB_Ypred)*100)

###ROC AND ROC AUC
#To check the performance of a clssification model
auc_NB = roc_auc_score(Y_test, NB_Ypred)
print("ROC_AUC_SCORE of navie-bayes classifier : ", auc_NB)
fpr_NB, tpr_NB, thresholds_NB = roc_curve(Y_test, NB_Ypred)
plt.plot(fpr_NB, tpr_NB, color='orange', label='ROC of navie-bayes classifier')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve of navie-bayes classifier (area=%0.2f')
plt.xlabel('False Positive Rate of navie-bayes classifier')
plt.ylabel('True positive Rate of navie-bayes classifier')
plt.title("Receiver Operating Characteristic(ROC) curve of navie-bayes classifier")
plt.legend()
plt.grid()
plt.show()

###############################################################
##################################################################
################################################################

###CONFUSION MATRIX OF "SVM(Support vector machine"
print("\n SVM ANALYSIS \n")
cmSV = confusion_matrix(Y_test, SV_Ypred)
print("\nConfusion matrix of SVM(Support Vector Machine) is \n",cmSV)
print("classification report of SVM(Support Vector Machine) : \n", classification_report(Y_test,SV_Ypred,digits=4))
TN_SV = cmSV[0,0]
print("True negative of SVM(Support Vector Machine) confusion matrix : ",TN_SV)
FP_SV = cmSV[0,1]
print("False positive of SVM(Support Vector Machine) confusion matrix : ",FP_SV)
FN_SV = cmSV[1,0]
print("False negative of SVM(Support Vector Machine) confusion matrix : ",FN_SV)
TP_SV = cmSV[1,1] 
print("True positive of SVM(Support Vector Machine) confusion matrix : ",TP_SV)
print("Accuracy rate of SVM(Support Vector Machine) {} ".format(np.divide(np.sum([cmSV[0,0],cmSV[1,1]]),np.sum(cmSV))*100))
print("Misclassification rate of SVM(Support Vector Machine):{}".format(np.divide(np.sum([cmSV[0,1],cmSV[1,0]]),np.sum(cmSV))*100))
table_SV = pd.crosstab(Y_test, SV_Ypred, rownames=['Actual Values'],colnames=['predicted values'], margins=True)
print("\n SVM(Support Vector Machine) cross table : ")
print(table_SV)
print("\n Accuracy score of SVM(Support Vector Machine) : ", accuracy_score(Y_test, SV_Ypred)*100)


###PRECISION(PPV-Positive predicted value)
precision_SV = TP_SV/(TP_SV+FP_SV)
precision_score_SV = TP_SV/float(TP_SV+FP_SV)*100
print('Precision score of SVM(Support Vector Machine):{0:0.4f}'.format(precision_score_SV))
###RECALL(TPR-True positive rate)
recall_score_SV = TP_SV/(TP_SV+FN_SV)
recall_score_SV = TP_SV/float(TP_SV+FN_SV)*100
print('recall or sensitivity score of SVM(Support Vector Machine):{0:0.4f}'.format(recall_score_SV))
###FALSE POSITIVE RATE(FPR)
fpr_SV = FP_SV/(FP_SV+TN_SV)
fpr_SV= FP_SV/float(FP_SV+TN_SV)*100
print('false positive rate of SVM(Support Vector Machine):{0:0.4f}'.format(fpr_SV))
###SPECIFICITY
specificity_SV = TN_SV/(TN_SV+FP_SV)
specificity_SV = TN_SV/float(TN_SV+FP_SV)*100
print('specificity of SVM(Support Vector Machine):{0:0.4}'.format(specificity_SV))
###F1_SCORE
print("f1_score of SVM(Support Vector Machine) is : ",f1_score(Y_test,SV_Ypred)*100)

###ROC AND ROC AUC
#To check the performance of a clssification model
auc_SV = roc_auc_score(Y_test, SV_Ypred)
print("ROC_AUC_SCORE of SVM(Support Vector Machine) : ", auc_SV)
fpr_SV, tpr_SV, thresholds_SV = roc_curve(Y_test, SV_Ypred)
plt.plot(fpr_SV, tpr_SV, color='orange', label='ROC of SVM(Support Vector Machine)')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve of SVM(Support Vector Machine) (area=%0.2f')
plt.xlabel('False Positive Rate of SVM(Support Vector Machine)')
plt.ylabel('True positive Rate of SVM(Support Vector Machine) ')
plt.title("Receiver Operating Characteristic(ROC) curve of SVM(Support Vector Machine)")
plt.legend()
plt.grid()
plt.show()

#############################################################
###############################################################
#############################################################

###CONFUSION MATRIX OF "decision tree"
print("\n DECISION TREE ANALYSIS \n")
cmDT = confusion_matrix(Y_test, DT_Ypred)
print("\nConfusion matrix of decision tree is \n",cmDT)
print("classification report of decision tree : \n", classification_report(Y_test,DT_Ypred,digits=4))
TN_DT = cmDT[0,0]
print("True negative of decision tree confusion matrix : ",TN_DT)
FP_DT = cmDT[0,1]
print("False positive of decision tree confusion matrix : ",FP_DT)
FN_DT = cmDT[1,0]
print("False negative of decision tree confusion matrix : ",FN_DT)
TP_DT = cmDT[1,1] 
print("True positive of decision tree confusion matrix : ",TP_DT)
print("Accuracy rate of decision tree {} ".format(np.divide(np.sum([cmDT[0,0],cmDT[1,1]]),np.sum(cmDT))*100))
print("Misclassification rate of decision tree:{}".format(np.divide(np.sum([cmDT[0,1],cmDT[1,0]]),np.sum(cmDT))*100))
table_DT = pd.crosstab(Y_test, DT_Ypred, rownames=['Actual Values'],colnames=['predicted values'], margins=True)
print("\n decision tree cross table :")
print(table_DT)
print("\n Accuracy score of decision tree : ", accuracy_score(Y_test, DT_Ypred)*100)


###PRECISION(PPV-Positive predicted value)
precision_DT = TP_DT/(TP_DT+FP_DT)
precision_score_DT = TP_DT/float(TP_DT+FP_DT)*100
print('Precision score of decision tree:{0:0.4f}'.format(precision_score_DT))
###RECALL(TPR-True positive rate)
recall_score_DT = TP_DT/(TP_DT+FN_DT)
recall_score_DT = TP_DT/float(TP_DT+FN_DT)*100
print('recall or sensitivity score of decision tree:{0:0.4f}'.format(recall_score_DT))
###FALSE POSITIVE RATE(FPR)
fpr_DT = FP_DT/(FP_DT+TN_DT)
fpr_DT= FP_DT/float(FP_DT+TN_DT)*100
print('false positive rate of decision tree:{0:0.4f}'.format(fpr_DT))
###SPECIFICITY
specificity_DT = TN_DT/(TN_DT+FP_DT)
specificity_DT = TN_DT/float(TN_DT+FP_DT)*100
print('specificity of decision tree:{0:0.4}'.format(specificity_DT))
###F1_SCORE
print("f1_score of decision tree is : ",f1_score(Y_test, DT_Ypred)*100)

###ROC AND ROC AUC
#To check the performance of a clssification model
auc_DT = roc_auc_score(Y_test, DT_Ypred)
print("ROC_AUC_SCORE of decision tree : ", auc_DT)
fpr_DT, tpr_DT, thresholds_DT = roc_curve(Y_test, DT_Ypred)
plt.plot(fpr_DT, tpr_DT, color='orange', label='ROC of decision tree')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve of decision tree (area=%0.2f')
plt.xlabel('False Positive Rate of decision tree')
plt.ylabel('True positive Rate of decision tree')
plt.title("Receiver Operating Characteristic(ROC) curve of decision tree")
plt.legend()
plt.grid()
plt.show()

#########################################################3
###################################################################
##################################################################

###CONFUSION MATRIX OF "random forest"
print("\n RANDOM FOREST ANALYSIS \n")
cmRF = confusion_matrix(Y_test, RF_Ypred)
print("\nConfusion matrix of random forest is ",cmRF)
print("classification report of random forest : \n", classification_report(Y_test,RF_Ypred,digits=4))
TN_RF = cmRF[0,0]
print("True negative of random forest confusion matrix : ",TN_RF)
FP_RF = cmRF[0,1]
print("False positive of random forest confusion matrix : ",FP_RF)
FN_RF = cmRF[1,0]
print("False negative of random forest confusion matrix : ",FN_RF)
TP_RF = cmRF[1,1] 
print("True positive of random forest confusion matrix : ",TP_RF)
print("Accuracy rate of random forest {} ".format(np.divide(np.sum([cmRF[0,0],cmRF[1,1]]),np.sum(cmRF))*100))
print("Misclassification rate of random forest:{}".format(np.divide(np.sum([cmRF[0,1],cmRF[1,0]]),np.sum(cmRF))*100))
table_RF = pd.crosstab(Y_test, RF_Ypred, rownames=['Actual Values'],colnames=['predicted values'], margins=True)
print("\n random forest cross table :")
print(table_RF)
print("\n Accuracy score of random forest : ", accuracy_score(Y_test, RF_Ypred)*100)


###PRECISION(PPV-Positive predicted value)
precision_RF = TP_RF/(TP_RF+FP_RF)
precision_score_RF = TP_RF/float(TP_RF+FP_RF)*100
print('Precision score of random forest:{0:0.4f}'.format(precision_score_RF))
###RECALL(TPR-True positive rate)
recall_score_RF = TP_RF/(TP_RF+FN_RF)
recall_score_RF = TP_RF/float(TP_RF+FN_RF)*100
print('recall or sensitivity score of random forest:{0:0.4f}'.format(recall_score_RF))
###FALSE POSITIVE RATE(FPR)
fpr_RF = FP_RF/(FP_RF+TN_RF)
fpr_RF= FP_RF/float(FP_RF+TN_RF)*100
print('false positive rate of random forest:{0:0.4f}'.format(fpr_RF))
###SPECIFICITY
specificity_RF = TN_RF/(TN_RF+FP_RF)
specificity_RF = TN_RF/float(TN_RF+FP_RF)*100
print('specificity of random forest:{0:0.4}'.format(specificity_RF))
###F1_SCORE
print("f1_score of random forest is : ",f1_score(Y_test, RF_Ypred)*100)

###ROC AND ROC AUC
#To check the performance of a clssification model
auc_RF = roc_auc_score(Y_test, RF_Ypred)
print("ROC_AUC_SCORE of random forest : ", auc_RF)
fpr_RF, tpr_RF, thresholds_RF = roc_curve(Y_test, RF_Ypred)
plt.plot(fpr_RF, tpr_RF, color='orange', label='ROC of random forest')
plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='ROC Curve of random forest (area=%0.2f')
plt.xlabel('False Positive Rate of random forest')
plt.ylabel('True positive Rate of random forest')
plt.title("Receiver Operating Characteristic(ROC) curve of random forest")
plt.legend()
plt.grid()
plt.show()

print('****************************')
print("Accuracy score of logistic regression is : ",accuracy_score(Y_test,LR_Ypred)*100)
print("Accuracy score of KNN is : ",accuracy_score(Y_test,KNN_Ypred)*100)
print("Accuracy score of naive-bayes is : ",accuracy_score(Y_test,NB_Ypred)*100)
print("Accuracy score of SVM is : ",accuracy_score(Y_test,SV_Ypred)*100)
print("Accuracy score of Decision tree is : ",accuracy_score(Y_test,DT_Ypred)*100)
print("Accuracy score of random forest is : ",accuracy_score(Y_test,RF_Ypred)*100)

import pickle 
pickle_out = open('Classifier.pkl', 'wb')
pickle.dump(LR, pickle_out)
pickle_out.close()
























