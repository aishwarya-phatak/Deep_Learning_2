from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Define actual and predicted labels (1 = Spam, 0 = Not Spam)
y_actual = [1,0,1,1,0,1,0,0,1,0]
print(np.shape(y_actual))
y_predicted = [1,0,1,0,0,1,1,0,1,0]
print(np.shape(y_predicted))

# 2. Compute confusion matrix
cm = confusion_matrix(y_actual, y_predicted)
print("Confusion Matrix:")
print(cm)

# 3. Extract TP, FP, FN, TN
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# 4. Print full metrics report
print("\nClassification Report:")
print(classification_report(y_actual, y_predicted))