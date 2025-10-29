from flask import Flask, request, jsonify
from preprocessing import Preprocessor
from modeltraining import ModelTrainer
from dataloader import DataLoader
from visualizer import Visualizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Carica i dati e addestra i modelli all'avvio
data_loader = DataLoader("/Users/ludovicamontanaro/progetto_titanic /src/src/dati/train.csv", "/Users/ludovicamontanaro/progetto_titanic /src/src/dati/test.csv")
train_set, test_set = data_loader.train_set, data_loader.test_set

preprocessor = Preprocessor(train_set, test_set)
preprocessor.preprocess()
train_set, test_set = preprocessor.train_set, preprocessor.test_set

# Definizione delle variabili di input e target
X = train_set.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y = train_set['Survived']

model_trainer = ModelTrainer(X, y)
X_valid, y_valid = model_trainer.train_models()
model_trainer.evaluate_models(X_valid, y_valid)

# Endpoint per ottenere previsioni
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Ottieni dati JSON dal corpo della richiesta
    
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Creazione di un DataFrame dai dati in input
    input_data = pd.DataFrame(data)
    
    # Preprobabilmente dovrai preprocessare anche qui i dati
    input_data = preprocessor.transform(input_data)  # Implementa questa funzione nella tua classe Preprocessor
    
    # Esegui le previsioni
    predictions = model_trainer.models['Logistic Regression'].predict(input_data)

    # Restituisci le previsioni in formato JSON
    return jsonify(predictions=predictions.tolist())

@app.route('/')
def home():
    return jsonify(message="Welcome to the Titanic Model API!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Rendi il server accessibile su tutte le interfacce