# model_evaluation.py
import pandas as pd
from sklearn.metrics import mean_absolute_error
import joblib
import logging

# Set up logging
logging.basicConfig(filename='model_evaluation.log', level=logging.INFO)

def evaluate_model(preprocessed_test_df, model):
    X_test = preprocessed_test_df.drop(columns=['Sales', 'Date', 'IsHoliday'])  # Assuming your test data also has these columns
    y_test = preprocessed_test_df['Sales']

    predictions = model.predict(X_test)
    loss = mean_absolute_error(y_test, predictions)
    
    logging.info(f'Mean Absolute Error: {loss}')
    return loss

if __name__ == "__main__":
    data_path = "C:\\Users\\b102western\\rossmann-sales-forecast\\data\\"
    model = joblib.load("rossmann_sales_model.pkl")
    
    # Load preprocessed test data
    preprocessed_test_df = pd.read_csv("preprocessed_test_df.csv")  # Placeholder for actual preprocessed test data
    
    evaluate_model(preprocessed_test_df, model)