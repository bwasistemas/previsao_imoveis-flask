import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. Carregamento e Preparação dos Dados ---
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name="MedHouseVal")

print("--- Amostra dos Dados ---")
print(X.head())
print("\n--- Variável Alvo ---")
print(y.head())
print("\n" + "="*30 + "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Criação da Pipeline de Machine Learning ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# --- 3. Treinamento do Modelo ---
print("Iniciando o treinamento da pipeline...")
pipeline.fit(X_train, y_train)
print("Treinamento concluído!")
print("\n" + "="*30 + "\n")

# --- 4. Realização de Previsões e Avaliação ---
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- 5. Exibição dos Resultados ---
print("--- Resultados da Avaliação do Modelo ---")
print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")
print("\n" + "="*30 + "\n")

print("--- Comparação: Valor Real vs. Previsão ---")
resultados = pd.DataFrame({'Valor Real': y_test, 'Valor Previsto': y_pred})
print(resultados.head(10).round(2))