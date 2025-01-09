# data_loading.py
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename='data_loading.log', level=logging.INFO)

def load_data(data_path):
    try:
        store_df = pd.read_csv(data_path + "store.csv")
        train_df = pd.read_csv(data_path + "train.csv")
        test_df = pd.read_csv(data_path + "test.csv")
        
        logging.info('Datasets loaded successfully')
        return store_df, train_df, test_df
    except Exception as e:
        logging.error(f'Error loading datasets: {e}')

if __name__ == "__main__":
    data_path = "C:\\Users\\b102western\\rossmann-sales-forecast\\data\\"
    store_df, train_df, test_df = load_data(data_path)
    print("Store, Train, and Test data loaded.")