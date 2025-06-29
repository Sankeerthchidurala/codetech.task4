import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))


# GUI App
def predict_species():
    try:
        values = [float(e1.get()), float(e2.get()), float(e3.get()), float(e4.get())]
        prediction = model.predict([values])
        species = iris.target_names[prediction[0]]
        messagebox.showinfo("Prediction Result", f"Predicted Iris Species: {species}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers in all fields.")

app = tk.Tk()
app.title("ML Model: Iris Flower Classifier")
app.geometry("400x350")

tk.Label(app, text="Enter Flower Features", font=("Arial", 16)).pack(pady=10)

frame = tk.Frame(app)
frame.pack()

tk.Label(frame, text="Sepal Length").grid(row=0, column=0)
e1 = tk.Entry(frame)
e1.grid(row=0, column=1)

tk.Label(frame, text="Sepal Width").grid(row=1, column=0)
e2 = tk.Entry(frame)
e2.grid(row=1, column=1)

tk.Label(frame, text="Petal Length").grid(row=2, column=0)
e3 = tk.Entry(frame)
e3.grid(row=2, column=1)

tk.Label(frame, text="Petal Width").grid(row=3, column=0)
e4 = tk.Entry(frame)
e4.grid(row=3, column=1)

tk.Button(app, text="Predict Species", command=predict_species, bg="green", fg="white").pack(pady=20)

tk.Label(app, text=f"Model Accuracy: {accuracy*100:.2f}%", font=("Arial", 10)).pack(pady=5)

app.mainloop()
