import os
import dill
import pandas as pd
import json
from datetime import datetime


def load_model(model_path):
    """Load the trained model from the specified file."""
    with open(model_path, 'rb') as file:
        model = dill.load(file)
    return model


def load_test_data(test_folder):
    """Load all test data files from the specified folder."""
    test_data = []
    for file_name in os.listdir(test_folder):
        if file_name.endswith('.json'):  # Assuming test files are in JSON format
            file_path = os.path.join(test_folder, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                test_data.append(data)
    return pd.DataFrame(test_data)


def make_predictions(model, data):
    """Make predictions using the loaded model."""
    predictions = model.predict(data)
    return predictions


def save_predictions(ids, predictions, output_folder):
    """Save predictions to a CSV file in the specified folder."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_file = os.path.join(output_folder, f'predictions_{timestamp}.csv')
    results = pd.DataFrame({
        'id': ids,
        'Predictions': predictions
    })
    results.to_csv(output_file, index=False)
    return output_file


def predict():
    """Main function to load model, make predictions, and save results."""
    model_path = os.path.join(os.getenv('PROJECT_PATH', '.'), 'data/models/cars_pipe_model.pkl')
    test_folder = os.path.join(os.getenv('PROJECT_PATH', '.'), 'data/test')
    output_folder = os.path.join(os.getenv('PROJECT_PATH', '.'), 'data/predictions')

    model = load_model(model_path)
    test_data = load_test_data(test_folder)

    ids = test_data['id']  # Extract IDs
    test_data = test_data.drop(columns=['id'])  # Drop IDs from test data

    print("Columns before prediction:", test_data.columns)  # Debugging statement
    predictions = make_predictions(model, test_data)
    output_file = save_predictions(ids, predictions, output_folder)
    print(f'Predictions saved to {output_file}')


if __name__ == "__main__":
    predict()
