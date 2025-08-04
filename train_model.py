import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv(url, names=columns)

# Step 2: Define X and y
X = df.drop("Outcome", axis=1)  # All columns except target
y = df["Outcome"]              # Target variable (0 or 1)

# Step 3: Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 4: Save model with pickle
with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved using ALL features!")


