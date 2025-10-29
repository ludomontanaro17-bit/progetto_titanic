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


class Preprocessor:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    def preprocess(self):
        self.fill_missing_values()
        self.feature_engineering()

    def fill_missing_values(self):
        self.train_set['Age'].fillna(self.train_set['Age'].median(), inplace=True)
        self.test_set['Age'].fillna(self.test_set['Age'].median(), inplace=True)
        self.train_set['Fare'].fillna(self.train_set['Fare'].median(), inplace=True)
        self.test_set['Fare'].fillna(self.test_set['Fare'].median(), inplace=True)

    def feature_engineering(self):
        self.train_set['Has_Cabin'] = self.train_set['Cabin'].notna().astype(int)
        self.test_set['Has_Cabin'] = self.test_set['Cabin'].notna().astype(int)
        self.train_set.drop('Cabin', axis=1, inplace=True)
        self.test_set.drop('Cabin', axis=1, inplace=True)
        self.train_set['FamilySize'] = self.train_set['SibSp'] + self.train_set['Parch'] + 1
        self.test_set['FamilySize'] = self.test_set['SibSp'] + self.test_set['Parch'] + 1
        self.train_set['Sex'] = self.train_set['Sex'].map({'male': 0, 'female': 1})
        self.test_set['Sex'] = self.test_set['Sex'].map({'male': 0, 'female': 1})
        self.train_set = pd.get_dummies(self.train_set, columns=['Embarked'], drop_first=True)
        self.test_set = pd.get_dummies(self.test_set, columns=['Embarked'], drop_first=True)

