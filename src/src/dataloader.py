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

class DataLoader:
    def __init__(self, train_path, test_path):
        self.train_set = pd.read_csv("/Users/ludovicamontanaro/progetto_titanic /src/src/train.csv")
        self.test_set = pd.read_csv("/Users/ludovicamontanaro/progetto_titanic /src/src/test.csv")


    def check_missing_values(self):
        return self.train_set.isnull().sum(), self.test_set.isnull().sum()
