import pandas as pd

 # Load the full dataset
full_df = pd.read_csv("KDDCup99.csv", header=None)  # Add column names later

# Show size of the full dataset
print("Full dataset shape:", full_df.shape)
# Step 2: Sample 10% of the data
sample_df = full_df.sample(frac=0.10, random_state=42)

 # the 10% sample to a new CSV file
sample_df.to_csv("kddcup_10percent.csv", index=False, header=False)

print("✅ 10% sample saved as 'kddcup_10percent.csv'")

import pandas as pd

# Define the KDD column names
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# Load the 10% dataset
df = pd.read_csv("kddcup_10percent.csv", names=column_names)

# Display some info
print("✅ 10% Dataset loaded.")
print("Shape:", df.shape)
print("First 5 rows:\n", df.head())

#preprocessing thedataset

from sklearn.preprocessing import LabelEncoder

# Step 1: Encode categorical features
categorical_cols = ["protocol_type", "service", "flag"]
le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Step 2: Convert label column to binary (normal = 0, attack = 1)
df["binary_label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

# Step 3: Drop original label if not needed
df.drop(columns=["label"], inplace=True)

# Step 4: Check for nulls
print("\nNull values per column:\n", df.isnull().sum())

# Final shape and preview
print("\n✅ Preprocessing complete. Final shape:", df.shape)
print(df.head())

# preprocessing finished 
# starting feature scaling 
from sklearn.preprocessing import MinMaxScaler

# Separate features and target
X = df.drop(columns=["binary_label"])
y = df["binary_label"]

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

print("✅ Feature scaling with MinMaxScaler complete.")
print("Scaled features shape:", X_scaled.shape)
## perfect cleaning 

##Random forest model building and testing 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Split dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_clf.fit(X_train, y_train)

# Predictions
y_pred = rf_clf.predict(X_test)
y_proba = rf_clf.predict_proba(X_test)[:, 1]

# Evaluation
print("=== Random Forest Classifier Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

### mashala lets see isolation forest 


from sklearn.linear_model import SGDClassifier

# Initialize Linear Classifier with default hinge loss (SVM-like)
linear_clf = SGDClassifier(loss='hinge', random_state=42, max_iter=1000, tol=1e-3)

# Train the model
linear_clf.fit(X_train, y_train)

# Predict
y_pred_linear = linear_clf.predict(X_test)
y_proba_linear = linear_clf.decision_function(X_test)  # for ROC AUC

# Evaluate
print("=== Linear Classifier (SGD) Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print("ROC AUC:", roc_auc_score(y_test, y_proba_linear))
print("\nClassification Report:\n", classification_report(y_test, y_pred_linear))

### KNN model 
# Reinitialize Isolation Forest with correct contamination level
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Initialize KNN (you can tune n_neighbors if needed)
knn_clf = KNeighborsClassifier(n_neighbors=5)

# Train
knn_clf.fit(X_train, y_train)

# Predict
y_pred_knn = knn_clf.predict(X_test)
y_proba_knn = knn_clf.predict_proba(X_test)[:, 1]

# Evaluate
print("=== K-Nearest Neighbors Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("ROC AUC:", roc_auc_score(y_test, y_proba_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

#### ploting the confusion matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Confusion matrices
cm_rf = confusion_matrix(y_test, rf_clf.predict(X_test))
cm_linear = confusion_matrix(y_test, linear_clf.predict(X_test))
cm_knn = confusion_matrix(y_test, knn_clf.predict(X_test))

titles = ["Random Forest", "Linear Classifier (SGD)", "K-Nearest Neighbors"]
cms = [cm_rf, cm_linear, cm_knn]

plt.figure(figsize=(18, 5))

for i, (cm, title) in enumerate(zip(cms, titles)):
    plt.subplot(1, 3, i+1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrices.png")
print("✅ Confusion matrices saved as 'confusion_matrices.png'")
#### plotting the classification report 

from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for headless saving
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get classification reports
report_rf = classification_report(y_test, rf_clf.predict(X_test), output_dict=True)
report_linear = classification_report(y_test, linear_clf.predict(X_test), output_dict=True)
report_knn = classification_report(y_test, knn_clf.predict(X_test), output_dict=True)

# Prepare data
models = ['Random Forest', 'Linear Classifier', 'KNN']
classes = ['0', '1']  # 0 = normal, 1 = attack
metrics = ['precision', 'recall', 'f1-score']
data = []

for model_name, report in zip(models, [report_rf, report_linear, report_knn]):
    for cls in classes:
        for metric in metrics:
            data.append({
                'Model': model_name,
                'Class': 'Normal' if cls == '0' else 'Attack',
                'Metric': metric.capitalize(),
                'Score': report[cls][metric]
            })

# Create DataFrame
df_metrics = pd.DataFrame(data)

# Plot
plt.figure(figsize=(14, 6))
sns.barplot(
    data=df_metrics,
    x='Metric',
    y='Score',
    hue='Model',
    ci=None,
    palette='Set2',
    edgecolor='gray'
)

plt.title("📊 Model Evaluation Metrics (Precision, Recall, F1-Score)")
plt.ylim(0.9, 1.01)  # Adjust if needed
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("model_metrics_comparison.png")
print("✅ Saved as 'model_metrics_comparison.png'")

### time comparison and execution time 

import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import accuracy_score

# === Measure and store model performance ===

performance_data = []

# --- Random Forest ---
start = time.time()
rf_clf.fit(X_train, y_train)
end = time.time()
acc_rf = accuracy_score(y_test, rf_clf.predict(X_test))
performance_data.append(["Random Forest", f"{acc_rf:.4f}", f"{end - start:.4f} sec"])

# --- Linear Classifier ---
start = time.time()
linear_clf.fit(X_train, y_train)
end = time.time()
acc_linear = accuracy_score(y_test, linear_clf.predict(X_test))
performance_data.append(["Linear Classifier", f"{acc_linear:.4f}", f"{end - start:.4f} sec"])

# --- K-Nearest Neighbors ---
start = time.time()
knn_clf.fit(X_train, y_train)
end = time.time()
acc_knn = accuracy_score(y_test, knn_clf.predict(X_test))
performance_data.append(["K-Nearest Neighbors", f"{acc_knn:.4f}", f"{end - start:.4f} sec"])

# === Plot table and save as image ===
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=performance_data,
    colLabels=["Model", "Accuracy", "Training Time"],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)

plt.title("📊 Model Accuracy & Training Time Comparison", fontsize=14, pad=10)
plt.tight_layout()
plt.savefig("model_performance_table.png")
print("✅ Table saved as 'model_performance_table.png'")


