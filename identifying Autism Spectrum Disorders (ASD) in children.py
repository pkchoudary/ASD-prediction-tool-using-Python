import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import messagebox


# Load Data
def load_data(path):
    data = pd.read_csv(path)
    print("Columns in dataset:", data.columns)  # Print column names to verify
    return data


# Preprocess Data
def preprocess_data(data):
    print("Initial columns:", data.columns)  # Print columns before preprocessing

    # Define columns to drop
    columns_to_drop = ['Qchat-10-Score', 'Who completed the test']

    # Drop columns if they exist
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    if existing_columns_to_drop:
        data = data.drop(columns=existing_columns_to_drop)

    # Convert categorical features to numeric
    data = pd.get_dummies(data, drop_first=True)

    print("Columns after preprocessing:", data.columns)  # Print columns after preprocessing

    # Check if target column exists
    target_column = 'Class/ASD Traits_Yes_Yes'
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found in the dataset.")

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler


# Create Model
def create_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)


# Train Model
def train_model(X_train, y_train):
    model = create_model()
    model.fit(X_train, y_train)
    return model


# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))


# Function to predict ASD based on user input
def predict_asd(user_input, model, scaler, data):
    input_df = pd.DataFrame([user_input], columns=data.columns[:-1])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]


# GUI Function
def create_gui(model, scaler, data):
    def submit():
        user_input = {
            'A1': var_a1.get(),
            'A2': var_a2.get(),
            'A3': var_a3.get(),
            'A4': var_a4.get(),
            'A5': var_a5.get(),
            'A6': var_a6.get(),
            'A7': var_a7.get(),
            'A8': var_a8.get(),
            'A9': var_a9.get(),
            'A10': var_a10.get(),
            'Age_Mons': var_age.get(),
            'Sex_m': 1 if var_sex.get() == 'Male' else 0,
            'Ethnicity_Latino': 1 if var_ethnicity.get() == 'Latino' else 0,
            'Ethnicity_Native Indian': 1 if var_ethnicity.get() == 'Native Indian' else 0,
            'Ethnicity_Others': 1 if var_ethnicity.get() == 'Others' else 0,
            'Ethnicity_Pacifica': 1 if var_ethnicity.get() == 'Pacifica' else 0,
            'Ethnicity_White European': 1 if var_ethnicity.get() == 'White European' else 0,
            'Ethnicity_asian': 1 if var_ethnicity.get() == 'Asian' else 0,
            'Ethnicity_black': 1 if var_ethnicity.get() == 'Black' else 0,
            'Ethnicity_middle eastern': 1 if var_ethnicity.get() == 'Middle Eastern' else 0,
            'Ethnicity_mixed': 1 if var_ethnicity.get() == 'Mixed' else 0,
            'Ethnicity_south asian': 1 if var_ethnicity.get() == 'South Asian' else 0,
            'Jaundice_yes': 1 if var_jaundice.get() == 'Yes' else 0,
            'Family_mem_with_ASD_yes': 1 if var_family.get() == 'Yes' else 0
        }

        try:
            prediction = predict_asd(user_input, model, scaler, data)
            if prediction == 1:
                messagebox.showinfo("Result", "The child might have Autism Spectrum Disorder (ASD).")
            else:
                messagebox.showinfo("Result", "The child might not have Autism Spectrum Disorder (ASD).")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Create the main window
    root = tk.Tk()
    root.title("ASD Prediction")

    # Define variables
    var_a1 = tk.IntVar()
    var_a2 = tk.IntVar()
    var_a3 = tk.IntVar()
    var_a4 = tk.IntVar()
    var_a5 = tk.IntVar()
    var_a6 = tk.IntVar()
    var_a7 = tk.IntVar()
    var_a8 = tk.IntVar()
    var_a9 = tk.IntVar()
    var_a10 = tk.IntVar()
    var_age = tk.IntVar()
    var_sex = tk.StringVar()
    var_ethnicity = tk.StringVar()
    var_jaundice = tk.StringVar()
    var_family = tk.StringVar()
    var_app = tk.StringVar()  # Add this line to define var_app

    # GUI Elements
    tk.Label(root, text="Answer the following questions (1=Yes, 0=No):").grid(row=0, column=0, columnspan=2)
    tk.Label(root, text="Q1: Does your child look at you when you call his/her name?").grid(row=1, column=0)
    tk.Entry(root, textvariable=var_a1).grid(row=1, column=1)

    tk.Label(root, text="Q2: Does your child point to indicate that he/she wants something?").grid(row=2, column=0)
    tk.Entry(root, textvariable=var_a2).grid(row=2, column=1)

    tk.Label(root, text="Q3: Does your child pretend play? (e.g. pretending to drink from an empty cup)").grid(row=3,
                                                                                                               column=0)
    tk.Entry(root, textvariable=var_a3).grid(row=3, column=1)

    tk.Label(root, text="Q4: Does your child follow your pointing to see something?").grid(row=4, column=0)
    tk.Entry(root, textvariable=var_a4).grid(row=4, column=1)

    tk.Label(root, text="Q5: Does your child imitate you? (e.g. if you make a face, does he/she copy it?)").grid(row=5,
                                                                                                                 column=0)
    tk.Entry(root, textvariable=var_a5).grid(row=5, column=1)

    tk.Label(root, text="Q6: Does your child respond to his/her name?").grid(row=6, column=0)
    tk.Entry(root, textvariable=var_a6).grid(row=6, column=1)

    tk.Label(root, text="Q7: Does your child make eye contact when he/she interacts with you?").grid(row=7, column=0)
    tk.Entry(root, textvariable=var_a7).grid(row=7, column=1)

    tk.Label(root, text="Q8: Does your child smile back at you when you smile at him/her?").grid(row=8, column=0)
    tk.Entry(root, textvariable=var_a8).grid(row=8, column=1)

    tk.Label(root, text="Q9: Does your child seem interested in other children?").grid(row=9, column=0)
    tk.Entry(root, textvariable=var_a9).grid(row=9, column=1)

    tk.Label(root, text="Q10: Does your child engage in pretend play with toys?").grid(row=10, column=0)
    tk.Entry(root, textvariable=var_a10).grid(row=10, column=1)

    tk.Label(root, text="Age in months:").grid(row=11, column=0)
    tk.Entry(root, textvariable=var_age).grid(row=11, column=1)

    tk.Label(root, text="Sex:").grid(row=12, column=0)
    tk.OptionMenu(root, var_sex, 'Male', 'Female').grid(row=12, column=1)

    tk.Label(root, text="Ethnicity:").grid(row=13, column=0)
    tk.OptionMenu(root, var_ethnicity, 'Latino', 'Native Indian', 'Others', 'Pacifica', 'White European', 'Asian',
                  'Black', 'Middle Eastern', 'Mixed', 'South Asian').grid(row=13, column=1)

    tk.Label(root, text="Jaundice (Yes/No):").grid(row=14, column=0)
    tk.Entry(root, textvariable=var_jaundice).grid(row=14, column=1)

    tk.Label(root, text="Family member with ASD (Yes/No):").grid(row=15, column=0)
    tk.Entry(root, textvariable=var_family).grid(row=15, column=1)

    tk.Label(root, text="Used screening app before (Yes/No):").grid(row=16, column=0)
    tk.Entry(root, textvariable=var_app).grid(row=16, column=1)

    tk.Button(root, text="Submit", command=submit).grid(row=17, column=0, columnspan=2)

    # Start the GUI loop
    root.mainloop()


if __name__ == "__main__":
    data_path = r"C:\Users\vpava\Downloads\Toddler Autism dataset July 20181.csv"  # Replace with your CSV file path
    data = load_data(data_path)
    (X_train, X_test, y_train, y_test), scaler = preprocess_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    create_gui(model, scaler, data)
