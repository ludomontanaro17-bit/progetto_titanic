from preprocessing import Preprocessor
from modeltraining import ModelTrainer
from dataloader import DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from visualizer import Visualizer

if __name__ == "__main__":
    # Caricamento dati
    data_loader = DataLoader("/Users/ludovicamontanaro/progetto_titanic /progetto_titanic/train.csv", "/Users/ludovicamontanaro/progetto_titanic /progetto_titanic/test.csv")
    train_set, test_set = data_loader.train_set, data_loader.test_set

    # Controllo valori mancanti
    missing_train, missing_test = data_loader.check_missing_values()
    print("Missing values in training set:\n", missing_train)
    print("Missing values in test set:\n", missing_test)

    # Preprocessing
    preprocessor = Preprocessor(train_set, test_set)
    preprocessor.preprocess()
    train_set, test_set = preprocessor.train_set, preprocessor.test_set

    # Definizione delle variabili di input e target
    X = train_set.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
    y = train_set['Survived']

    # Training dei modelli
    model_trainer = ModelTrainer(X, y)
    X_valid, y_valid = model_trainer.train_models()
    model_trainer.evaluate_models(X_valid, y_valid)

    # Visualizzazione della curva ROC per ciascun modello
    for name, model in model_trainer.models.items():
        y_pred_prob = model.predict_proba(X_valid)[:, 1]
        visualizer = Visualizer()
        visualizer.plot_roc_curve(y_valid, y_pred_prob, name)

########################