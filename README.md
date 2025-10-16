# MultiLabel-x-HierarquicoLocalNoPai


⚙️ Parâmetros do XGBoost

Os parâmetros utilizados no treinamento dos modelos foram obtidos via Grid Search, executado sobre as 10 partições (folds) de treino.
Após a validação cruzada, foi selecionada a melhor combinação de hiperparâmetros, que apresentou o melhor equilíbrio entre desempenho e generalização.
Essa configuração foi aplicada tanto na abordagem Multilabel Global quanto na Hierárquica Local por Nó Pai.

🔹 Parâmetros finais selecionados
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",        
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

📘 Descrição dos parâmetros
Parâmetro
objective	Define o tipo de problema. O valor "binary:logistic" indica uma tarefa de classificação binária, retornando probabilidades entre 0 e 1.
eval_metric	Métrica usada durante o treinamento para avaliar a performance do modelo. O "logloss" mede o erro de previsão probabilística.
max_depth	Profundidade máxima das árvores. Controla a complexidade do modelo — valores menores evitam sobreajuste (overfitting).
learning_rate	Taxa de aprendizado (também conhecida como eta). Define o peso de cada nova árvore adicionada ao modelo. Valores menores tornam o aprendizado mais estável.
n_estimators	Número total de árvores (iterações) geradas no processo de boosting.
subsample	Proporção de amostras do conjunto de treino usada em cada árvore. Reduz sobreajuste ao introduzir variabilidade.
colsample_bytree	Proporção de colunas (features) usadas em cada árvore. Também ajuda a reduzir sobreajuste e aumentar a robustez.


| Label              | Multilabel | Hierárquico |
| :----------------- | :--------: | :---------: |
| Hate.speech        |   0.5566   |    0.3561   |
| Sexism             |   0.5950   |    0.4518   |
| Body               |   0.7536   |    0.7695   |
| Racism             |   0.0869   |    0.2627   |
| Ideology           |   0.0945   |    0.1817   |
| Homophobia         |   0.4806   |    0.3371   |
| Origin             |   0.0000   |    0.0000   |
| Religion           |   0.1000   |    0.1671   |
| OtherLifestyle     |   0.0000   |    0.0000   |
| Fat.people         |   0.7238   |    0.7677   |
| Left.wing.ideology |   0.0000   |    0.0250   |
| Ugly.people        |   0.6962   |    0.7591   |
| Black.people       |   0.0952   |    0.1961   |
| Fat.women          |   0.7199   |    0.7747   |
| Feminists          |   0.1202   |    0.2656   |
| Gays               |   0.1393   |    0.1015   |
| Immigrants         |   0.1333   |    0.1333   |
| Islamists          |   0.0000   |    0.0000   |
| Lesbians           |   0.5597   |    0.3306   |
| Men                |   0.0722   |    0.2399   |
| Muslims            |   0.1000   |    0.1067   |
| Refugees           |   0.1936   |    0.3783   |
| Trans.women        |   0.1333   |    0.0500   |
| Women              |   0.5873   |    0.4317   |
| Transexuals        |   0.0000   |    0.1000   |
| Ugly.women         |   0.6793   |    0.7293   |
| Migrants           |   0.1980   |    0.3864   |
| Homossexuals       |   0.5172   |    0.3282   |


<img width="1400" height="600" alt="graph" src="https://github.com/user-attachments/assets/ade496f0-12f0-4d3b-9004-7703a5c9ce32" />
