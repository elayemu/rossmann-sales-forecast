# data_visualization.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(filename='data_visualization.log', level=logging.INFO)

def plot_sales_distribution(train_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['Sales'], kde=True)
    plt.title('Sales Distribution (Train Data)')
    plt.show()

def plot_holiday_effect(train_df):
    train_df['IsHoliday'] = train_df['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)
    holiday_sales = train_df.groupby('IsHoliday')['Sales'].mean()
    
    holiday_sales.plot(kind='bar')
    plt.title('Sales on Holidays vs Non-Holidays')
    plt.ylabel('Average Sales')
    plt.xlabel('Is Holiday')
    plt.show()

def plot_promo_effect(train_df):
    promo_effect = train_df.groupby('Promo')['Sales'].mean()
    promo_effect.plot(kind='bar')
    plt.title('Effect of Promotions on Sales')
    plt.ylabel('Average Sales')
    plt.xlabel('Promo Active')
    plt.show()

if __name__ == "__main__":
    data_path = "C:\\Users\\b102western\\rossmann-sales-forecast\\data\\"
    store_df = pd.read_csv(data_path + "store.csv")
    train_df = pd.read_csv(data_path + "train.csv")

    plot_sales_distribution(train_df)
    plot_holiday_effect(train_df)
    plot_promo_effect(train_df)