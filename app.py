import numpy as np
from flask import Flask, request, render_template, jsonify
import joblib
import os

# --- Importações do Modelo ---
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --- Configuração do App Flask ---
app = Flask(__name__)
MODEL_PATH = 'modelo_random_forest.joblib'

# --- Lógica do Modelo ---
def treinar_e_salvar_modelo():
    """Treina o modelo RandomForest e o salva em um arquivo."""
    print("Iniciando o treinamento do modelo RandomForest...")
    
    # 1. Carregar dados
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # 2. Dividir em treino e teste
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Criar a pipeline com RandomForest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # 4. Treinar o modelo
    pipeline.fit(X_train, y_train)
    
    # 5. Salvar a pipeline (que inclui o scaler e o regressor)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo treinado e salvo em {MODEL_PATH}")
    return pipeline

def carregar_modelo():
    """Carrega o modelo do arquivo, ou treina se o arquivo não existir."""
    if not os.path.exists(MODEL_PATH):
        return treinar_e_salvar_modelo()
    else:
        print(f"Carregando modelo existente de {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

# Carrega o modelo ao iniciar a aplicação
modelo = carregar_modelo()

# --- Rotas da Aplicação Web ---
@app.route('/')
def home():
    """Renderiza a página inicial com o formulário."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recebe os dados do formulário, faz a previsão e mostra o resultado."""
    try:
        # 1. Coletar dados do formulário
        # Os nomes devem ser exatamente os mesmos do formulário HTML
        features = [
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveBedrms']),
            float(request.form['Population']),
            float(request.form['AveOccup']),
            float(request.form['Latitude']),
            float(request.form['Longitude'])
        ]
        
        # 2. Converter para o formato que o modelo espera (array 2D)
        dados_para_prever = np.array(features).reshape(1, -1)
        
        # 3. Fazer a previsão
        # A pipeline cuida da padronização e da predição
        prediction_raw = modelo.predict(dados_para_prever)[0]
        
        # 4. Formatar o resultado
        # O target está em centenas de milhares de dólares
        valor_estimado = f"{prediction_raw * 100000:,.2f}"
        
        # 5. Renderizar a página novamente, mas agora com o resultado
        return render_template('index.html', prediction=valor_estimado)

    except Exception as e:
        # Em caso de erro, retorna uma mensagem clara
        return render_template('index.html', prediction=f"Erro ao processar: {e}")

# --- Ponto de Entrada da Aplicação ---
if __name__ == '__main__':
    # Inicia o servidor web do Flask
    # host='0.0.0.0' torna o servidor acessível na sua rede local
    app.run(host='0.0.0.0', port=5000, debug=True)
