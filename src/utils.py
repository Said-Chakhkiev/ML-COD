import pandas as pd
from sklearn.preprocessing import StandardScaler 

def load_data(file_path):
    """Загрузка данных из CSV файла"""
    return pd.read_csv(file_path)

def preprocess_data(data, training=True):
    """Предобработка данных: масштабирование и разделение на X и y"""
    X = data.drop(columns=['failure'])  
    y = data['failure'] if 'failure' in data.columns else None

    scaler = StandardScaler()
    X = scaler.fit_transform(X) if training else scaler.transform(X)

    return X, y
