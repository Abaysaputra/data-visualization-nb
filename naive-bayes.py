import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Variabel global untuk menyimpan dataset, model, dan label encoder
dataset = None
model = None
label_encoder = None
normalized_probabilities = None

# Fungsi untuk membaca dataset dari file CSV
def load_dataset():
    global dataset
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if filepath:
        try:
            dataset = pd.read_csv(filepath)
            text_box.insert(tk.END, f"Dataset Loaded:\n{dataset.head()}\n")
        except Exception as e:
            text_box.insert(tk.END, f"Error loading dataset: {e}\n")

# Fungsi untuk menghitung prior probabilities
def calculate_prior_probabilities():
    global prior_probabilities
    if dataset is not None and "X_LAMA_PEROLEH_KERJAAN" in dataset.columns:
        prior_probabilities = dataset["X_LAMA_PEROLEH_KERJAAN"].value_counts(normalize=True).to_dict()
        # Visualisasi prior probabilities
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(prior_probabilities.keys()), y=list(prior_probabilities.values()), palette="viridis")
        plt.title("Prior Probabilities (P(H))", fontsize=14)
        plt.xlabel("Classes", fontsize=12)
        plt.ylabel("Probability", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        text_box.insert(tk.END, f"Prior Probabilities (P(H)):\n{prior_probabilities}\n")
    else:
        text_box.insert(tk.END, "Kolom 'X_LAMA_PEROLEH_KERJAAN' tidak ditemukan dalam dataset.\n")

# Fungsi untuk menghitung likelihoods (kondisi probabilitas)
def calculate_likelihoods():
    global likelihoods
    target = "X_LAMA_PEROLEH_KERJAAN"
    
    if dataset is not None and target in dataset.columns:
        likelihoods = {}
        features = dataset.drop(columns=[target])
        
        for class_value in dataset[target].unique():
            likelihoods[class_value] = {}
            subset = dataset[dataset[target] == class_value]
            for col in features.columns:
                likelihoods[class_value][col] = subset[col].value_counts(normalize=True).to_dict()

        # Visualisasi likelihoods untuk setiap fitur dan kelas
        for feature in features.columns:
            likelihood_data = []
            for class_value in dataset[target].unique():
                class_likelihood = likelihoods[class_value].get(feature, {})
                for feature_value, prob in class_likelihood.items():
                    likelihood_data.append({
                        "Class": class_value,
                        "Feature Value": feature_value,
                        "Probability": prob
                    })

            likelihood_df = pd.DataFrame(likelihood_data)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=likelihood_df, x="Feature Value", y="Probability", hue="Class", palette="muted")
            plt.title(f"Likelihoods for Feature '{feature}'", fontsize=14)
            plt.xlabel(f"{feature} Values", fontsize=12)
            plt.ylabel("Likelihood (P(E|H))", fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

        text_box.insert(tk.END, f"\nLikelihoods (P(E|H)) calculated successfully.\n")
    else:
        text_box.insert(tk.END, "Kolom 'X_LAMA_PEROLEH_KERJAAN' tidak ditemukan dalam dataset.\n")

# Fungsi untuk prediksi menggunakan model Naive Bayes
def make_prediction():
    global model, label_encoder, normalized_probabilities
    if dataset is not None and "X_LAMA_PEROLEH_KERJAAN" in dataset.columns:
        X = dataset.drop(columns=["X_LAMA_PEROLEH_KERJAAN"])
        y = dataset["X_LAMA_PEROLEH_KERJAAN"]
        
        # Encoding fitur categorical
        global label_encoder
        label_encoder = LabelEncoder()
        for col in X.select_dtypes(include='object').columns:
            X[col] = label_encoder.fit_transform(X[col])
        
        model = CategoricalNB()
        model.fit(X, y)

        # Prediksi dan perhitungan posterior probabilities
        probabilities = model.predict_proba(X)
        normalized_probabilities = dict(zip(model.classes_, probabilities.mean(axis=0)))

        predicted_class = max(normalized_probabilities, key=normalized_probabilities.get)
        text_box.insert(tk.END, f"\nPredicted Class: {predicted_class}\n")
        text_box.insert(tk.END, f"Posterior Probabilities:\n{tabulate(normalized_probabilities.items(), headers=['Class', 'Probability'])}\n")
    else:
        text_box.insert(tk.END, "Kolom 'X_LAMA_PEROLEH_KERJAAN' tidak ditemukan dalam dataset.\n")



# Fungsi untuk visualisasi posterior probabilities
def visualize_posterior_probabilities():
    global normalized_probabilities
    if 'normalized_probabilities' in globals() and normalized_probabilities is not None:
        # Visualisasi posterior probabilities
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(normalized_probabilities.keys()), y=list(normalized_probabilities.values()), palette="coolwarm")
        plt.title("Posterior Probabilities (P(H|E))", fontsize=14)
        plt.xlabel("Classes", fontsize=12)
        plt.ylabel("Normalized Probability", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        text_box.insert(tk.END, "Posterior probabilities visualized.\n")
    else:
        text_box.insert(tk.END, "Normalized probabilities belum dihitung.\n")

# Membuat antarmuka pengguna (GUI)
root = tk.Tk()
root.title("Naive Bayes Predictor")

# Frame untuk kontrol
frame_controls = tk.Frame(root)
frame_controls.pack(padx=10, pady=10, fill="x")

btn_load = tk.Button(frame_controls, text="Load Dataset", command=load_dataset)
btn_load.pack(side="left")

btn_prior = tk.Button(frame_controls, text="Calculate Prior Probabilities", command=calculate_prior_probabilities)
btn_prior.pack(side="left")

btn_likelihood = tk.Button(frame_controls, text="Calculate Likelihoods", command=calculate_likelihoods)
btn_likelihood.pack(side="left")

btn_predict = tk.Button(frame_controls, text="Make Prediction", command=make_prediction)
btn_predict.pack(side="left")

btn_visualize = tk.Button(frame_controls, text="Visualize Posterior Probabilities", command=visualize_posterior_probabilities)
btn_visualize.pack(side="left")

# Text box untuk menampilkan pesan dan hasil
text_box = scrolledtext.ScrolledText(root, height=15, width=80, wrap="word")
text_box.pack(padx=10, pady=10, fill="both", expand=True)

root.mainloop()
