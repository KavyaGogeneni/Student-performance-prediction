import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("student.csv")

print("===== DATASET =====")
print(data)

X = data[['StudyHours', 'Attendance', 'PreviousMarks']]

y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("\n===== PREDICTIONS =====")
print(prediction)

accuracy = accuracy_score(y_test, prediction)

print("\n===== ACCURACY =====")
print(accuracy * 100)

print("\n===== NEW STUDENT PREDICTION =====")

study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance: "))
previous_marks = float(input("Enter Previous Marks: "))

new_data = pd.DataFrame(
    [[study_hours, attendance, previous_marks]],
    columns=['StudyHours', 'Attendance', 'PreviousMarks']
)

result = model.predict(new_data)

print("\nFINAL RESULT =", result[0])