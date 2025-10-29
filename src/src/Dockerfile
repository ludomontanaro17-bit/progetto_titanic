# Utilizzare un'immagine base di Python
FROM python:3.9-slim


# Copiare i requisiti del progetto
COPY requirements.txt .

# Installare le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Copiare l'intero progetto
COPY . .

# Esporre la porta del container
EXPOSE 5000

# Comando per eseguire lo script principale
CMD ["python", "main.py"]  