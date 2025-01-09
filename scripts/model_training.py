# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Set up logging
logging.basicConfig(filename='model_training.log', level=logging.INFO)

def train_model(preprocessed_train_df):
    X = preprocessed_train_df.drop(columns=['Sales', 'Date', 'IsHoliday'])  # Exclude date and target variable
    y = preprocessed_train_df['Sales']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a model pipeline
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor())
    ])
    model_pipeline.fit(X_train, y_train)

    # Save the model
    model_file = "rossmann_sales_model.pkl"
    joblib.dump(model_pipeline, model_file)
    logging.info(f'Model saved as {model_file}')

if __name__ == "__main__":
    data_path = "C:\\Users\\b102western\\rossmann-sales-forecast\\data\\"
    store_df = pd.read_csv(data_path + "store.csv")
    train_df = pd.read_csv(data_path + "train.csv")
    
    preprocessed_train_df = pd.read_csv("preprocessed_train_df.csv")  # Assuming you save the preprocessed data
    train_model(preprocessed_train_df)