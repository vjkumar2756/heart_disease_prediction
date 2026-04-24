import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # Handle zero values in Cholesterol
    if 'Cholesterol' in df.columns:
        mean_val = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
        df['Cholesterol'] = df['Cholesterol'].replace(0, mean_val)

    # Handle zero values in RestingBP
    if 'RestingBP' in df.columns:
        mean_val = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
        df['RestingBP'] = df['RestingBP'].replace(0, mean_val)

    return df


def encode_data(df):
    df = pd.get_dummies(df, drop_first=True)
    return df.astype(int)


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)   # IMPORTANT (no fit here)
    return X_train_scaled, X_test_scaled, scaler


def split_data(df, target_col='HeartDisease'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y