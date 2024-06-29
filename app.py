import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import os

# Charger le modèle LightGBM
model= joblib.load("C:\\Users\\amal9\\OneDrive\\Documents\\9-Openclassroom\\8-PROJET 7\\GIT\\lightgbm_model_df1.pkl")

df = pd.read_csv("C:\\Users\\amal9\\OneDrive\\Documents\\9-Openclassroom\\8-PROJET 7\\GIT\\df1_final.csv")

app = Flask(__name__)

@app.route('/predict_score', methods=['GET'])
def predict_score():
    
    try:
        # Récupérez les données d'entrée au format JSON depuis la requête
        data = request.get_json(force=True)
        print(data)
       #Assurez-vous que les données reçues correspondent aux caractéristiques attendues par le modèle
        input_data = data.get('data')
        
        df[df['SK_ID_CURR']==input_data]
        print(input_data)
        if input_data is None:
            return jsonify({'error': 'Missing data field'})

# Sélectionnez les données du DataFrame correspondant à l'identifiant SK_ID_CURR spécifié
        client = df[df['SK_ID_CURR'] == input_data]

# Vérifiez si des données ont été trouvées pour l'identifiant donné
        if client.empty:
            return jsonify({'error': 'SK_ID_CURR not found'})

# Utilisez les données sélectionnées pour la prédiction
        y_pred_classes = model.predict_proba(client.values)[:, 1]  # Obtenez la probabilité de la classe positive

# Convertissez le tableau NumPy en une liste Python avant de le renvoyer dans la réponse JSON
        y_pred_classes_list = y_pred_classes.tolist()
# Retournez les scores prédits sous forme de liste
        return jsonify({'prediction': y_pred_classes_list})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8887)