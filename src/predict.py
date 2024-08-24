import yaml 
import joblib 
import pandas as pd 
from utils import preprocess_data

with open('src/config.yaml') as f:
    config = yaml.safe_load(f)

def predict(input_file, output_file):
    model = joblib.load(config['model']['model_path'])
    data = pd.read_csv(input_file)
    X, _ = preprocess_data(data, training=False)

    predictions = model.predict(X)
    data['prediction'] = predictions
    data.to_csv(output_file, index=False)
    print(f"Предсказания сохранены в {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Предсказание выхода из строя дисков")
    parser.add_argument("--input", required=True, help="Путь к файлу с входными данными")
    parser.add_argument("--output", required=True, help="Путь для сохранения предсказаний")
    args = parser.parse_args()
    predict(args.input, args.output)
