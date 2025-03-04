import numpy as np
import pandas as pd


def load_dataset(filepath="data/perceptron_dataset.csv"):
    """Načte dataset a vypíše základní info."""
    df = pd.read_csv(filepath)
    print("Rozměry datasetu:", df.shape)
    return df


def prepare_features_labels(df):
    """Připraví matici příznaků X a cílovou proměnnou y. Přidá sloupec 1 pro bias."""
    X = df[['Temperature', 'Light', 'CO2']].values
    y = df['Occupancy'].values
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    return X, y


def split_dataset(df, train_ratio=0.8, random_state=42):
    """Rozdělí data na trénovací a testovací množinu se zachováním poměru tříd."""
    class0 = df[df['Occupancy'] == 0].sample(frac=1, random_state=random_state)
    class1 = df[df['Occupancy'] == 1].sample(frac=1, random_state=random_state)
    
    split_0 = int(train_ratio * len(class0))
    split_1 = int(train_ratio * len(class1))
    
    train_df = pd.concat([class0.iloc[:split_0], class1.iloc[:split_1]], ignore_index=True)
    test_df  = pd.concat([class0.iloc[split_0:], class1.iloc[split_1:]], ignore_index=True)
    
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print("Počet trénovacích vzorků:", train_df.shape[0])
    print("Počet testovacích vzorků:", test_df.shape[0])
    
    return train_df, test_df


def get_numpy_data(train_df, test_df):
    """Převede data z DataFrame do NumPy pole a přidá sloupec 1 pro bias."""
    X_train = train_df[['Temperature', 'Light', 'CO2']].values
    y_train = train_df['Occupancy'].values
    X_test  = test_df[['Temperature', 'Light', 'CO2']].values
    y_test  = test_df['Occupancy'].values

    X_train = np.hstack([np.ones((X_train.shape[0],1)), X_train])
    X_test  = np.hstack([np.ones((X_test.shape[0],1)), X_test])
    
    return X_train, y_train, X_test, y_test
