# MultiLabel-x-HierarquicoLocalNoPai


⚙️ Parâmetros do XGBoost

Os parâmetros utilizados no treinamento dos modelos foram obtidos via Grid Search, executado sobre as 10 partições (folds) de treino.
Após a validação cruzada, foi selecionada a melhor combinação de hiperparâmetros, que apresentou o melhor equilíbrio entre desempenho e generalização.
Essa configuração foi aplicada tanto na abordagem Multilabel Global quanto na Hierárquica Local por Nó Pai.

`🔹 Parâmetros finais selecionados`
`XGB_PARAMS = {`
    `"objective": "binary:logistic",`
    `"eval_metric": "logloss",`        
    `"max_depth": 4,`
    `"learning_rate": 0.1,`
    `"n_estimators": 200,`
    `"subsample": 0.8,`
    `"colsample_bytree": 0.8`
`}`


| 🧩 Parâmetro         | 💡 Função principal                                                                                                            |
| :------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| **objective**        | Define o tipo de problema. <br>→ `"binary:logistic"` indica classificação binária, retornando probabilidades entre 0 e 1.      |
| **eval_metric**      | Métrica de avaliação usada no treino. <br>→ `"logloss"` mede o erro entre a probabilidade prevista e o rótulo real.            |
| **max_depth**        | Profundidade máxima das árvores. <br>→ Controla a complexidade do modelo e evita *overfitting*.                                |
| **learning_rate**    | Taxa de aprendizado. <br>→ Define o quanto cada nova árvore influencia o modelo final.                                         |
| **n_estimators**     | Número de árvores (iterações) no *boosting*. <br>→ Mais árvores aumentam a capacidade do modelo, mas também o tempo de treino. |
| **subsample**        | Proporção de amostras usadas por árvore. <br>→ Introduz variabilidade e reduz *overfitting*.                                   |
| **colsample_bytree** | Proporção de colunas (features) usadas por árvore. <br>→ Aumenta a diversidade entre as árvores e melhora a generalização.     |


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

------------------------------------------------------------------------------------------------------------------------------------



🔧 Hiperparâmetros Otimizados (Optuna)

| Parâmetro          | Valor                 | Descrição breve                                   |
| ------------------ | --------------------- | ------------------------------------------------- |
| `objective`        | `binary:logistic`     | Função objetivo para classificação binária        |
| `eval_metric`      | `logloss`             | Métrica de avaliação usada durante o treino       |
| `max_depth`        | `4`                   | Profundidade máxima das árvores                   |
| `learning_rate`    | `0.145192689133182`   | Taxa de aprendizado                               |
| `n_estimators`     | `440`                 | Número de árvores (iterações do boosting)         |
| `subsample`        | `0.5054945946218856`  | Proporção de amostras usadas em cada árvore       |
| `colsample_bytree` | `0.747819692180028`   | Proporção de colunas usadas em cada árvore        |
| `gamma`            | `1.1676055677106392`  | Penalização por divisão de nó (reduz overfitting) |
| `min_child_weight` | `6`                   | Peso mínimo da soma de instâncias em um nó filho  |
| `reg_lambda`       | `0.31512291092719386` | Regularização L2                                  |
| `reg_alpha`        | `0.03389975701017645` | Regularização L1                                  |

<img width="1400" height="600" alt="optuna" src="https://github.com/user-attachments/assets/fc80b6de-3bec-4ac3-bbf2-c56d503c080b" />


------------------------------------------------------------------------------------------------------------------------------------



🧩 Melhores Parâmetros por Fold — XGBoost
| Fold | `max_depth` | `learning_rate` | `n_estimators` | `subsample` | `colsample_bytree` |  `mean_f1` |
| :--: | :---------: | :-------------: | :------------: | :---------: | :----------------: | :--------: |
|   1  |      4      |       0.10      |       200      |     1.0     |         0.8        | **0.2818** |
|   2  |      6      |       0.10      |       200      |     0.8     |         0.8        | **0.2811** |
|   3  |      6      |       0.10      |       200      |     1.0     |         1.0        | **0.2795** |
|   4  |      8      |       0.10      |       200      |     0.8     |         0.8        | **0.2776** |
|   5  |      4      |       0.05      |       200      |     0.8     |         1.0        | **0.2754** |
|   6  |      4      |       0.05      |       200      |     1.0     |         0.8        | **0.2745** |
|   7  |      8      |       0.10      |       200      |     1.0     |         0.8        | **0.2738** |
|   8  |      4      |       0.05      |       200      |     1.0     |         1.0        | **0.2734** |
|   9  |      4      |       0.10      |       100      |     1.0     |         0.8        | **0.2728** |
|  10  |      8      |       0.10      |       100      |     0.8     |         1.0        | **0.2698** |

<img width="1400" height="600" alt="gridSearchFold" src="https://github.com/user-attachments/assets/e0201299-e4de-4a9f-9914-a6841146f20e" />

