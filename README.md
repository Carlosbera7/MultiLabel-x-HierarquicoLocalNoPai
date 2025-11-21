# MultiLabel-x-HierarquicoLocalNoPai


⚙️ Parâmetros do XGBoost

Os parâmetros utilizados no treinamento dos modelos foram obtidos via Grid Search, executado sobre as 10 partições (folds) de treino.
Após a validação cruzada, foi selecionada a melhor combinação de hiperparâmetros, que apresentou o melhor equilíbrio entre desempenho e generalização.
Essa configuração foi aplicada tanto na abordagem Multilabel Global quanto na Hierárquica Local por Nó Pai.



| 🧩 Parâmetro         | 💡 Função principal                                                                                                            |
| :------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| **objective**        | Define o tipo de problema. <br>→ `"binary:logistic"` indica classificação binária, retornando probabilidades entre 0 e 1.      |
| **eval_metric**      | Métrica de avaliação usada no treino. <br>→ `"logloss"` mede o erro entre a probabilidade prevista e o rótulo real.            |
| **max_depth**        | Profundidade máxima das árvores. <br>→ Controla a complexidade do modelo e evita *overfitting*.                                |
| **learning_rate**    | Taxa de aprendizado. <br>→ Define o quanto cada nova árvore influencia o modelo final.                                         |
| **n_estimators**     | Número de árvores (iterações) no *boosting*. <br>→ Mais árvores aumentam a capacidade do modelo, mas também o tempo de treino. |
| **subsample**        | Proporção de amostras usadas por árvore. <br>→ Introduz variabilidade e reduz *overfitting*.                                   |
| **colsample_bytree** | Proporção de colunas (features) usadas por árvore. <br>→ Aumenta a diversidade entre as árvores e melhora a generalização.     |



------------------------------------------------------------------------------------------------------------------------------------
Optuna:

| Hiperparâmetro       | Faixa / Valores Possíveis | Tipo de Busca                                 |
| -------------------- | ------------------------- | --------------------------------------------- |
| **max_depth**        | 3 → 10                    | Inteiro (`suggest_int`)                       |
| **learning_rate**    | 0.01 → 0.3                | Float logarítmico (`suggest_float`, log=True) |
| **n_estimators**     | 100 → 500                 | Inteiro (`suggest_int`)                       |
| **subsample**        | 0.5 → 1.0                 | Float contínuo (`suggest_float`)              |
| **colsample_bytree** | 0.5 → 1.0                 | Float contínuo (`suggest_float`)              |

🔧 Hiperparâmetros Otimizados (Optuna)

| Fold | max_depth | learning_rate       | n_estimators | subsample          | colsample_bytree   | 
| ---- | --------- | ------------------- | ------------ | ------------------ | ------------------ | 
| 1    | 6         | 0.0798983053167349  | 475          | 0.8396055777138034 | 0.9625632338704708 | 
| 2    | 3         | 0.2914704979601401  | 448          | 0.84937152114332   | 0.7848596544349612 | 
| 3    | 9         | 0.20605478040269745 | 302          | 0.5887203371202743 | 0.8323104402558865 | 
| 4    | 7         | 0.2036723012822396  | 274          | 0.5175241685295008 | 0.8610036062835934 | 
| 5    | 6         | 0.21885672889855487 | 194          | 0.968375051179664  | 0.7317999002183174 | 
| 6    | 3         | 0.2914704979601401  | 448          | 0.84937152114332   | 0.7848596544349612 | 
| 7    | 3         | 0.2731869021241011  | 465          | 0.7044624801022293 | 0.7343526094439211 | 
| 8    | 3         | 0.12146048889211235 | 462          | 0.6727610834615692 | 0.7958561724249666 | 
| 9    | 9         | 0.14298926156497807 | 175          | 0.5340144691485039 | 0.7215243003652387 | 
| 10   | 6         | 0.07235055697929252 | 477          | 0.7527880068719619 | 0.7275690660180654 | 



optuna por fold : 
<img width="1400" height="600" alt="optuna por fold" src="https://github.com/user-attachments/assets/1aa0ce76-3fd8-42b7-bfaa-c0dcaebbae5d" />

  

Melhor parametro Geral : 

objective": "binary:logistic",
    "eval_metric": "logloss",        
    "max_depth": 4,
    "learning_rate":  0.145192689133182,
    "n_estimators": 440,
    "subsample": 0.5054945946218856,
    "colsample_bytree": 0.747819692180028,

    
<img width="1400" height="600" alt="optuna geral" src="https://github.com/user-attachments/assets/a1a087a2-5899-41de-aaad-48ecd4f699d5" />

------------------------------------------------------------------------------------------------------------------------------------
Grid Search :

| Hiperparâmetro       | Valores Possíveis       |
| -------------------- | ----------------------- |
| **max_depth**        | 3, 6, 10                |
| **learning_rate**    | 0.01, 0.1, 0.3          |
| **n_estimators**     | 100, 200, 300, 400, 500 |
| **subsample**        | 0.5, 1.0                |
| **colsample_bytree** | 0.5, 1.0                |



🧩 Melhores Parâmetros por Fold — XGBoost
| Fold | max_depth | learning_rate | n_estimators | subsample | colsample_bytree | mean_f1             |
| ---- | --------- | ------------- | ------------ | --------- | ---------------- | ------------------- |
| 1    | 3         | 0.1           | 500          | 0.5       | 0.5              | 0.2990094785368235  |
| 2    | 6         | 0.3           | 400          | 0.5       | 0.5              | 0.29352211856573573 |
| 3    | 10        | 0.3           | 200          | 0.5       | 0.5              | 0.2924494023498848  |
| 4    | 10        | 0.1           | 200          | 0.5       | 1.0              | 0.28273535436860764 |
| 5    | 6         | 0.3           | 100          | 1.0       | 1.0              | 0.2761988171837599  |
| 6    | 6         | 0.1           | 300          | 1.0       | 0.5              | 0.27551604605231167 |
| 7    | 10        | 0.1           | 300          | 1.0       | 0.5              | 0.2746221761815351  |
| 8    | 10        | 0.01          | 400          | 1.0       | 0.5              | 0.23394632683290095 |
| 9    | 3         | 0.01          | 500          | 1.0       | 0.5              | 0.229114418682963   |
| 10   | 6         | 0.01          | 300          | 1.0       | 0.5              | 0.22553299717460945 |


<img width="1400" height="600" alt="gridSearchFoldNew" src="https://github.com/user-attachments/assets/dfc4d645-662d-4b7a-87dd-531708ec84c5" />


PArametrização Grid Search GEral :
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",        
    "max_depth": 3,
    "learning_rate":  0.1,
    "n_estimators": 500,
    "subsample": 0.5,
    "colsample_bytree": 0.5
}

<img width="1400" height="600" alt="grid_search_geral" src="https://github.com/user-attachments/assets/b08c3c1a-12b6-4585-b75d-61f4d1f2df75" />




| Classe             | Multilabel (Optuna) | Hierárquico (Optuna) | Multilabel (GridSearch) | Hierárquico (GridSearch) | Multilabel (Optuna Geral) | Hierárquico (Optuna Geral) | Multilabel (GridSearch Geral) | Hierárquico (GridSearch Geral) |
| ------------------ | ------------------: | -------------------: | ----------------------: | -----------------------: | ------------------------: | -------------------------: | ----------------------------: | -----------------------------: |
| Black.people       |            0.033333 |             0.189003 |                0.153571 |                 0.185799 |                  0.173016 |                   0.191604 |                      0.120238 |                       0.228435 |
| Body               |            0.733289 |             0.767723 |                0.734328 |                 0.729750 |                  0.710624 |                   0.741927 |                      0.763421 |                       0.771668 |
| Fat.people         |            0.690047 |             0.764617 |                0.702403 |                 0.727286 |                  0.685177 |                   0.739740 |                      0.730064 |                       0.769098 |
| Fat.women          |            0.696795 |             0.766263 |                0.706346 |                 0.723673 |                  0.690187 |                   0.710696 |                      0.729372 |                       0.763110 |
| Feminists          |            0.145238 |             0.216007 |                0.156602 |                 0.200413 |                  0.194365 |                   0.110681 |                      0.141667 |                       0.242007 |
| Gays               |            0.132143 |             0.127879 |                0.139286 |                 0.100716 |                  0.135714 |                   0.137029 |                      0.132143 |                       0.141082 |
| Hate.speech        |            0.541421 |             0.541421 |                0.563487 |                 0.563487 |                  0.590624 |                   0.590624 |                      0.573518 |                       0.573518 |
| Homophobia         |            0.444841 |             0.371409 |                0.505877 |                 0.308055 |                  0.540447 |                   0.290268 |                      0.537750 |                       0.327061 |
| Homossexuals       |            0.478694 |             0.366807 |                0.501555 |                 0.301356 |                  0.543404 |                   0.274264 |                      0.556691 |                       0.319040 |
| Ideology           |            0.073333 |             0.162695 |                0.085618 |                 0.151854 |                  0.138951 |                   0.116480 |                      0.113030 |                       0.171504 |
| Immigrants         |            0.066667 |             0.166667 |                0.066667 |                 0.066667 |                  0.000000 |                   0.000000 |                      0.133333 |                       0.133333 |
| Islamists          |            0.000000 |             0.106667 |                0.000000 |                 0.155556 |                  0.000000 |                   0.028571 |                      0.000000 |                       0.066667 |
| Left.wing.ideology |            0.000000 |             0.000000 |                0.000000 |                 0.000000 |                  0.000000 |                   0.000000 |                      0.000000 |                       0.042222 |
| Lesbians           |            0.518246 |             0.377535 |                0.551797 |                 0.304095 |                  0.569319 |                   0.270647 |                      0.587818 |                       0.317818 |
| Men                |            0.111667 |             0.200905 |                0.132222 |                 0.215201 |                  0.172475 |                   0.185919 |                      0.091667 |                       0.254337 |
| Migrants           |            0.222323 |             0.348830 |                0.197807 |                 0.380267 |                  0.301405 |                   0.352431 |                      0.317199 |                       0.409059 |
| Muslims            |            0.100000 |             0.050000 |                0.100000 |                 0.050000 |                  0.000000 |                   0.000000 |                      0.100000 |                       0.106667 |
| Origin             |            0.000000 |             0.000000 |                0.000000 |                 0.000000 |                  0.000000 |                   0.000000 |                      0.000000 |                       0.000000 |
| OtherLifestyle     |            0.000000 |             0.000000 |                0.000000 |                 0.066667 |                  0.000000 |                   0.000000 |                      0.000000 |                       0.000000 |
| Racism             |            0.093333 |             0.252206 |                0.163800 |                 0.259246 |                  0.197729 |                   0.218131 |                      0.139184 |                       0.272842 |
| Refugees           |            0.216919 |             0.375196 |                0.206032 |                 0.375196 |                  0.266638 |                   0.344116 |                      0.251235 |                       0.409210 |
| Religion           |            0.050000 |             0.178333 |                0.050000 |                 0.185476 |                  0.083333 |                   0.140000 |                      0.090000 |                       0.195087 |
| Sexism             |            0.578073 |             0.461772 |                0.594783 |                 0.453496 |                  0.592927 |                   0.441209 |                      0.611079 |                       0.457894 |
| Trans.women        |            0.133333 |             0.050000 |                0.116667 |                 0.050000 |                  0.000000 |                   0.000000 |                      0.116667 |                       0.083333 |
| Transexuals        |            0.000000 |             0.000000 |                0.000000 |                 0.000000 |                  0.000000 |                   0.000000 |                      0.000000 |                       0.000000 |
| Ugly.people        |            0.677110 |             0.763622 |                0.679860 |                 0.702461 |                  0.648667 |                   0.709927 |                      0.685411 |                       0.759711 |
| Ugly.women         |            0.657170 |             0.697669 |                0.633477 |                 0.697669 |                  0.598803 |                   0.647963 |                      0.655939 |                       0.723706 |
| Women              |            0.579154 |             0.442695 |                0.589971 |                 0.445623 |                  0.624596 |                   0.443187 |                      0.624771 |                       0.450174 |
| **MÉDIA**          |          **0.2817** |           **0.3261** |              **0.2931** |               **0.3087** |                **0.2982** |                 **0.2793** |                    **0.3097** |                     **0.3198** |

Para ambas as abordagens de classificação, plana e hierárquica, os seguintes hiperparâmetros do \texttt{XGBoost} foram mantidos fixos:

    objective: binary:logistic (função objetivo para tarefas de classificação binária)
    eval_metric: logloss (métrica de avaliação por log-loss)
    device: cuda (execução em GPU)

A otimização concentrou-se nos parâmetros abaixo, conforme os intervalos definidos:

    n_estimators: 100 a 1000 (número máximo de árvores a serem criadas no modelo)
    learning_rate: 0,001 e 0,3 (escala logarítmica)
    max_depth: 1 a 10 (profundidade máxima da árvore)
    subsample: 0,05 a 1,0 (proporção de amostras por árvore)
    colsample_bytree: 0,05 a 1,0 (fração de atributos por árvore)
    min_child_weight: 1 a 20 (mínimo de instâncias por nó folha)

| Label                | F1 Médio (Local no Pai) | F1 Médio (Multilabel) |
|---------------------|------------------------:|----------------------:|
| Fat.women           | 0.793026               | 0.712968              |
| Body                | 0.774056               | 0.737845              |
| Fat.people          | 0.771810               | 0.713508              |
| Ugly.people         | 0.764381               | 0.682302              |
| Ugly.women          | 0.731110               | 0.657494              |
| Sexism              | 0.476666               | 0.595224              |
| Women               | 0.451467               | 0.606187              |
| Refugees            | 0.436596               | 0.254013              |
| Migrants            | 0.429644               | 0.211190              |
| Homophobia          | 0.403977               | 0.501557              |
| Lesbians            | 0.401788               | 0.555606              |
| Homossexuals        | 0.396994               | 0.506226              |
| Hate.speech         | 0.558165               | 0.558165              |
| Racism              | 0.279572               | 0.099487              |
| Men                 | 0.228970               | 0.094444              |
| Black.people        | 0.220044               | 0.061905              |
| Feminists           | 0.206405               | 0.145238              |
| Trans.women         | 0.166667               | 0.133333              |
| Religion            | 0.163420               | 0.100000              |
| Ideology            | 0.161116               | 0.071515              |
| Gays                | 0.143824               | 0.110714              |
| Immigrants          | 0.133333               | 0.133333              |
| Muslims             | 0.106667               | 0.100000              |
| Transexuals         | 0.100000               | 0.000000              |
| Left.wing.ideology  | 0.065000               | 0.000000              |
| Islamists           | 0.000000               | 0.000000              |
| OtherLifestyle      | 0.000000               | 0.000000              |
| Origin              | 0.000000               | 0.000000              |
| **Média Geral**     | **0.334454**            | **0.308972**          |



