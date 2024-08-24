import yaml 
import joblib 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from utils import load_data, preprocess_data

with open('src/config.yaml') as f:
    config = yaml.safe_load(f)

def train_model():
    data = load_data(config['data']['raw_data_path'])
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['model']['test_size'], random_state=config['model']['random_state'])

    model = RandomForestClassifier(random_state=config['model']['random_state'])
    model.fit(X_train, y_train)

    joblib.dump(model, config['model']['model_path'])
    print("Модель успешно обучена и сохранена.")

if __name__ == "__main__":
    train_model()
