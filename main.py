from src.data_preprocessing import load_data, clean_data, encode_data, split_data, scale_data
from src.model_training import train_models
from src.evaluation import evaluate_all
from sklearn.model_selection import train_test_split

# 1. Load
df = load_data("data/heart.csv") 

# 2. Clean
df = clean_data(df)

# 3. Encode
df = encode_data(df)

# 4. Split X, y
X, y = split_data(df)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Scale
X_train, X_test, scaler = scale_data(X_train, X_test)

# 7. Train models
models = train_models(X_train, y_train)

# 8. Evaluate
results = evaluate_all(models, X_test, y_test)

# 9. Print results
for model_name, metrics in results.items():
    print(f"\n🔹 {model_name}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

import pickle

# choose best model (example: Logistic Regression)
best_model = models["LogisticRegression"]

# create folder if not exists
import os
os.makedirs("models", exist_ok=True)

# save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(best_model, f)
