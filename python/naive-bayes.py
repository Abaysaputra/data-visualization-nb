import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Fungsi untuk membaca dataset
def load_dataset():
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        global dataset
        try:
            dataset = pd.read_csv(filepath)
            text_box.insert(tk.END, f"Dataset Loaded:\n{dataset.head()}\n")
            text_box.insert(tk.END, f"\nColumns in dataset: {list(dataset.columns)}\n")
        except Exception as e:
            text_box.insert(tk.END, f"Error loading dataset: {e}\n")

# Fungsi untuk menghitung prior probabilities
def calculate_prior_probabilities():
    global priors
    if "X_LAMA_PEROLEH_KERJAAN" in dataset.columns:
        priors = dataset["X_LAMA_PEROLEH_KERJAAN"].value_counts(normalize=True).to_dict()
        # Tampilkan prior probabilities menggunakan Treeview untuk format yang lebih rapi
        prior_window = tk.Toplevel(root)
        prior_window.title("Prior Probabilities")
        tree = ttk.Treeview(prior_window, columns=("Class", "Probability"), show="headings")
        tree.heading("Class", text="Class")
        tree.heading("Probability", text="Probability")

        for cls, prob in priors.items():
            tree.insert("", tk.END, values=(cls, prob))

        tree.pack(padx=10, pady=10)
        text_box.insert(tk.END, f"\nPrior Probabilities (P(H)):\n{priors}\n")
    else:
        text_box.insert(tk.END, "Column 'X_LAMA_PEROLEH_KERJAAN' not found in dataset.\n")

# Fungsi untuk menghitung likelihoods
def calculate_likelihoods():
    global likelihoods
    target = "X_LAMA_PEROLEH_KERJAAN"
    
    if target in dataset.columns:
        likelihoods = {}
        features = dataset.drop(columns=[target])
        
        for class_value in dataset[target].unique():
            likelihoods[class_value] = {}
            subset = dataset[dataset[target] == class_value]
            for col in features.columns:
                likelihoods[class_value][col] = subset[col].value_counts(normalize=True).to_dict()

        # Tampilkan likelihoods menggunakan Treeview untuk format yang lebih rapi
        likelihood_window = tk.Toplevel(root)
        likelihood_window.title("Likelihoods")
        tree = ttk.Treeview(likelihood_window, columns=("Class", "Feature", "Value", "Probability"), show="headings")
        tree.heading("Class", text="Class")
        tree.heading("Feature", text="Feature")
        tree.heading("Value", text="Value")
        tree.heading("Probability", text="Probability")

        for cls, feats in likelihoods.items():
            for feat, vals in feats.items():
                for val, prob in vals.items():
                    tree.insert("", tk.END, values=(cls, feat, val, prob))

        tree.pack(padx=10, pady=10)
        text_box.insert(tk.END, "\nLikelihoods (P(E|H)):\n")
    else:
        text_box.insert(tk.END, "Column 'X_LAMA_PEROLEH_KERJAAN' not found in dataset.\n")

# Fungsi untuk prediksi
def make_prediction():
    input_data = {}
    for field, entry in user_inputs.items():
        input_value = entry.get().strip()
        
        # Pengecekan apakah kolom 'X_PRODI' dan lainnya ada dalam dataset
        if field not in dataset.columns:
            text_box.insert(tk.END, f"Input untuk '{field}' tidak ditemukan dalam dataset.\n")
            return
        
        # Validasi input data
        if input_value == '':
            text_box.insert(tk.END, f"Input untuk '{field}' tidak boleh kosong.\n")
            return
        
        # Cek apakah input untuk 'X_PRODI' adalah nilai yang valid (sesuai kategori)
        if field == "X_PRODI":
            try:
                # Convert input to integer using LabelEncoder
                input_value = LabelEncoder.transform([input_value])[0]
            except ValueError:
                text_box.insert(tk.END, f"Nilai tidak valid untuk '{field}'.\n")
                return
        
        input_data[field] = input_value
    
    # Prediksi menggunakan model
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        class_probabilities = model.predict_proba(input_df)[0]
        normalized_probabilities = {f"{cls}": prob for cls, prob in zip(model.classes_, class_probabilities)}

        # Display results in a structured manner
        result_window = tk.Toplevel(root)
        result_window.title("Prediction Results")
        tree = ttk.Treeview(result_window, columns=("Class", "Probability"), show="headings")
        tree.heading("Class", text="Class")
        tree.heading("Probability", text="Probability")

        for cls, prob in normalized_probabilities.items():
            tree.insert("", tk.END, values=(cls, prob))

        tree.pack(padx=10, pady=10)
        text_box.insert(tk.END, f"\nPredicted Class: {prediction}\n")
    except Exception as e:
        text_box.insert(tk.END, f"Error during prediction: {e}\n")

# Fungsi untuk validasi model
def validate_model():
    global model
    if "X_LAMA_PEROLEH_KERJAAN" in dataset.columns:
        X = dataset.drop(columns=["X_LAMA_PEROLEH_KERJAAN"])
        y = dataset["X_LAMA_PEROLEH_KERJAAN"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = CategoricalNB()
        model.fit(X_train, y_train)

        scores = cross_val_score(model, X, y, cv=5)
        text_box.insert(tk.END, f"\nCross-Validation Scores: {scores}\n")
        text_box.insert(tk.END, f"Mean Score: {scores.mean()}\n")

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=y.unique())
        text_box.insert(tk.END, f"\nClassification Report:\n{report}\n")
    else:
        text_box.insert(tk.END, "Column 'X_LAMA_PEROLEH_KERJAAN' not found in dataset.\n")

# Fungsi untuk visualisasi data
def visualize_data():
    # Plot prior probabilities
    if 'priors' in globals():
        classes = list(priors.keys())
        probabilities = list(priors.values())
        plt.figure(figsize=(10, 5))
        plt.bar(classes, probabilities, color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Probability')
        plt.title('Prior Probabilities (P(H))')
        plt.show()

    # Plot likelihoods
    if 'likelihoods' in globals():
        for cls, feats in likelihoods.items():
            plt.figure(figsize=(12, 6))
            for feat, vals in feats.items():
                values = list(vals.keys())
                probabilities = list(vals.values())
                plt.bar(values, probabilities, alpha=0.7, label=f"Class {cls} - Feature {feat}")
            plt.xlabel('Values')
            plt.ylabel('Probability')
            plt.title(f"Likelihoods (P(E|H)) for Class {cls}")
            plt.legend()
            plt.show()

# Membuat antarmuka pengguna
root = tk.Tk()
root.title("Naive Bayes Predictor")

# Frame untuk kontrol
frame_controls = tk.Frame(root)
frame_controls.pack(pady=10)

btn_load = tk.Button(frame_controls, text="Load Dataset", command=load_dataset)
btn_load.pack(side=tk.LEFT, padx=5)

btn_prior = tk.Button(frame_controls, text="Calculate Priors", command=calculate_prior_probabilities)
btn_prior.pack(side=tk.LEFT, padx=5)

btn_likelihoods = tk.Button(frame_controls, text="Calculate Likelihoods", command=calculate_likelihoods)
btn_likelihoods.pack(side=tk.LEFT, padx=5)

btn_validate = tk.Button(frame_controls, text="Validate Model", command=validate_model)
btn_validate.pack(side=tk.LEFT, padx=5)

btn_predict = tk.Button(frame_controls, text="Make Prediction", command=make_prediction)
btn_predict.pack(side=tk.LEFT, padx=5)

# Button untuk visualisasi data
btn_visualize = tk.Button(frame_controls, text="Visualize Data", command=visualize_data)
btn_visualize.pack(side=tk.LEFT, padx=5)

# Frame untuk input user
frame_inputs = tk.LabelFrame(root, text="Input Data for Prediction")
frame_inputs.pack(pady=10, fill="x", padx=10)

user_inputs = {}
for feature in ["X_PRODI", "X_KELAMIN", "X_KELAS", "X_KONSENTRASI", "X_LAMA_STUDI", "X_IPK"]:
    lbl = tk.Label(frame_inputs, text=feature)
    lbl.pack(anchor="w", padx=5)
    entry = tk.Entry(frame_inputs)
    entry.pack(fill="x", padx=5, pady=2)
    user_inputs[feature] = entry

# Frame untuk output teks
frame_output = tk.Frame(root)
frame_output.pack(pady=10, padx=10, fill="both", expand=True)

text_box = scrolledtext.ScrolledText(frame_output, wrap=tk.WORD, width=80, height=20)
text_box.pack(pady=5, padx=5, fill="both", expand=True)

root.mainloop()
