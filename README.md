# previsao_imoveis-flask

## Mais Informações

### Como o Modelo Funciona

Este projeto utiliza um modelo de Regressão de Floresta Aleatória (`RandomForestRegressor`) para prever o valor de imóveis na Califórnia. O fluxo de trabalho do modelo é o seguinte:

1.  **Carregamento dos Dados**: O modelo é treinado com o dataset "California Housing", que contém informações sobre imóveis na Califórnia.
2.  **Pré-processamento**: Os dados são padronizados usando `StandardScaler` para que todas as features tenham a mesma escala.
3.  **Treinamento**: Um modelo `RandomForestRegressor` é treinado com os dados. Este modelo é um conjunto de árvores de decisão que trabalham juntas para fazer uma previsão mais precisa.
4.  **Pipeline**: O pré-processamento e o modelo são combinados em uma `Pipeline`, o que garante que os novos dados para previsão passem pelas mesmas etapas de transformação dos dados de treinamento.
5.  **Previsão**: A aplicação Flask recebe os dados do usuário através de um formulário, os processa através da pipeline e retorna o valor previsto para o imóvel.

O modelo treinado é salvo no arquivo `modelo_random_forest.joblib` para que não precise ser treinado novamente a cada vez que a aplicação é iniciada.

---

*Desenvolvido por: Bruno Winicius Amorim - FIAP*
