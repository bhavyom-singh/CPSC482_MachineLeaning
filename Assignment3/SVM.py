import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

#loading data
all_data = pd.read_csv("Assignment3/Data/emails.csv")

#splitting data into testing and training
test_size = 0.25
X_train, X_test, Y_train, Y_test = train_test_split(all_data.text, all_data.spam, test_size = test_size)

#counting occurence of each word in training values
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)

#creating model
model = SVC(probability=True)

#training model
model.fit(X_train_count, Y_train)

X_test_count = v.transform(X_test)

#accuracy
score = model.score(X_test_count, Y_test)
print("SVM score : {}".format(score))

#confusion matix
Y_pred = model.predict(X_test_count)
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion matrix : {}".format(confusion_matrix))

#Sensitivity, Specificity, Precision
TN = confusion_matrix[0,0]
TP = confusion_matrix[1,1]
FN = confusion_matrix[1,0]
FP = confusion_matrix[0,1]

print("Sensitivity : {}".format(TP/(TP+FN)))
print("Specificity : {}".format(TN/(TN+FP)))
print("Precision : {}".format(TP/(TP+FP)))

# ROC and area under the curve
Y_pred_prob = model.predict_proba(X_test_count)[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
print("Area under the curve = {}".format(auc(x = fpr, y = tpr)))

##Confidence Interval of error

## Using confusion matrix
# n = (TN + TP + FN + FP)
# print("n = {}".format(n))
# error_H = (FN + FP)/n
# print("error_h = {}".format(error_H))
# confidence_plus = error_H + 1.96 * (np.sqrt(error_H * (1 - error_H) / n))
# confidence_minus = error_H - 1.96 * (np.sqrt(error_H * (1 - error_H) / n)) 
# print("Confidence Interval range using fn,fp from {:.2f}% to {:.2f}%".format((confidence_plus * 100), confidence_minus * 100))

## Using accuracy
## error = 1 - accuracy
error_s_H = 1 - score
confidence_plus_s = error_s_H + 1.96 * (np.sqrt(error_s_H * (1 - error_s_H) / (len(all_data)* test_size)))
confidence_minus_s = error_s_H - 1.96 * (np.sqrt(error_s_H * (1 - error_s_H) / (len(all_data)*test_size))) 
print("Confidence Interval of Error using score from {:.2f}% to {:.2f}%".format((confidence_plus_s * 100), confidence_minus_s * 100))


#ploting the ROC Curve
plt.plot(fpr, tpr, label = 'ROC Curve')
plt.plot([0,1],[0,1], 'k--', label = 'Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.legend(loc = "lower right")
plt.show()