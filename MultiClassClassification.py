# Mohammed Ahmed Zakiuddin
# 1001675091
# To run the code python MultiClassClassification.py

import pandas as pd
from reportlab.pdfgen import canvas
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Load dataset
glass = pd.read_csv('glass.txt', sep=',', header=None)

# Split dataset into features and target
X = glass.iloc[:, :-1]
y = glass.iloc[:, -1]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Predict on the test set
y_pred = rfc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Initialize the PDF document
pdf_name = "output.pdf"
pdf = canvas.Canvas(pdf_name)

pdf.drawString(50, 800, "Name: Mohammed Ahmed Zakiuddin")
pdf.drawString(50, 770, "ID: 1001675091")
pdf.drawString(50, 740, "Algorithm: Random Forest Classifier from scikit-learn")
pdf.drawString(50, 700, "Training accuracy: {}".format(rfc.score(X_train, y_train)))
pdf.drawString(50, 670, "Testing accuracy: {}".format(accuracy))

# Predict the classes of the given inputs
inputs = [[1.51761, 12.81, 3.54, 1.23, 73.24, 0.58, 8.39, 0.00, 0.00],
          [1.51514, 14.01, 2.68, 3.50, 69.89, 1.68, 5.87, 2.20, 0.00],
          [1.51937, 13.79, 2.41, 1.19, 72.76, 0.00, 9.77, 0.00, 0.00],
          [1.51658, 14.80, 0.00, 1.99, 73.11, 0.00, 8.28, 1.71, 0.00]]

class_names = {1: "building windows", 
               2: "housing windows",
               3: "vehicle windows",
               4: "trucking windows",
               5: "containers",
               6: "tableware",
               7: "headlamps"}

pdf.drawString(50, 620, "Predictions for the given inputs:")
pdf.drawString(50, 590, "-"*50)

for i, input_ in enumerate(inputs):
    prediction = rfc.predict([input_])
    pdf.drawString(50, 560 - i*30, "Input {}: {}".format(i+1, class_names[prediction[0]]))

pdf.save()
print("Results saved to {}".format(pdf_name))
