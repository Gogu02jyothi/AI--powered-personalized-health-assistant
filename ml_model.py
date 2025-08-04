import pickle

# Load model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# Take user input
print("Please enter the following health data:")

pregnancies = int(input("Pregnancies: "))
glucose = int(input("Glucose Level: "))
bp = int(input("Blood Pressure: "))
skin = int(input("Skin Thickness (mm): "))
insulin = int(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

# Format input
features = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]

# Predict
prediction = model.predict(features)

# Show result
if prediction[0] == 1:
    print("You are at **High Risk** of Diabetes.")
else:
    print("You are at **Low Risk** of Diabetes.")
