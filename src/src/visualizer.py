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


class Visualizer:
    @staticmethod
    def plot_roc_curve(y_valid, y_pred_prob, model_name):
        fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')  # Linea di riferimento
        plt.title(f'Curva ROC - {model_name}')
        plt.xlabel('Tasso di Falsi Positivi')
        plt.ylabel('Tasso di Veri Positivi')
        plt.legend(loc='best')
        plt.show()