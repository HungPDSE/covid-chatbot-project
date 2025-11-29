import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

def load_and_preprocess_data(filepath):
    """Lấy dữ liệu từ file csv"""
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values(by=["location", "date"]).reset_index(drop=True)
    le = LabelEncoder()
    df['location_id'] = le.fit_transform(df['location'])

    df['new_cases_log'] = np.log1p(df['new_cases'])
    df['new_deaths_log'] = np.log1p(df['new_deaths'])
    df['vaccinations_log'] = np.log1p(df['new_vaccinations_smoothed'])
    scaler_minmax = MinMaxScaler()
    scaler_robust = RobustScaler()
    df['stringency_scaled'] = scaler_minmax.fit_transform(df[['stringency_index']])
    df['vaccinated_scaled'] = scaler_robust.fit_transform(df[['people_fully_vaccinated_per_hundred']])
    return df, le

def create_sequences(df, features, timesteps=7):
    """Tạo chuỗi thời gian cho model Bi-LSTM"""
    X_seq, X_location, y = [], [], []
    for location_id, group in df.groupby('location_id'):
        group = group.dropna(subset=features + ['new_cases_log']).reset_index(drop=True)
        values = group[features].values
        for i in range(len(values) - timesteps):
            X_seq.append(values[i:i+timesteps])
            y.append(group['new_cases_log'].iloc[i+timesteps])
            X_location.append(location_id)
    X_seq = np.array(X_seq)
    X_location = np.array(X_location).reshape(-1, 1)
    y = np.array(y)
    return X_seq, X_location, y

def train_test_split_by_country(X_seq, X_location, y, test_size=0.2):
    """Tách data train và test theo từng quốc gia"""
    unique_ids = np.unique(X_location)
    train_idx, test_idx = [], []
    for uid in unique_ids:
        idx = np.where(X_location.flatten() == uid)[0]
        cutoff = int((1 - test_size) * len(idx))
        train_idx.extend(idx[:cutoff])
        test_idx.extend(idx[cutoff:])
    X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]
    X_train_country, X_test_country = X_location[train_idx], X_location[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train_seq, X_test_seq, X_train_country, X_test_country, y_train, y_test
