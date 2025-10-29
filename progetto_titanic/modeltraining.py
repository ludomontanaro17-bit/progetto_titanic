from preprocessing import Preprocessor
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

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = {}
    
    def train_models(self):
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Logistic Regression
        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = model_lr

        # Random Forest
        model_rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        model_rf.fit(X_train, y_train)
        self.models['Random Forest'] = model_rf

        # XGBoost
        model_xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
        model_xgb.fit(X_train, y_train)
        self.models['XGBoost'] = model_xgb

        return X_valid, y_valid

    def evaluate_models(self, X_valid, y_valid):
        for name, model in self.models.items():
            y_pred = model.predict(X_valid)
            accuracy = accuracy_score(y_valid, y_pred)
            report = classification_report(y_valid, y_pred)
            print(f"{name} - Accuratezza: {accuracy}\n{report}")

         # Confusion Matrix
        conf_matrix = confusion_matrix(y_valid, y_pred)
        self.visualize_confusion_matrix(conf_matrix, name)
    
    def visualize_confusion_matrix(self, conf_matrix, model_name):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non Sopravvissuti', 'Sopravvissuti'],
                    yticklabels=['Non Sopravvissuti', 'Sopravvissuti'])
        plt.title(f'Matrice di Confusione - {model_name}')
        plt.xlabel('Predizioni')
        plt.ylabel('Reali')
        plt.show()