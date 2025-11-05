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



| Classe             | Multilabel (Optuna) | Hierárquico (Optuna) | Multilabel (GridSearch) | Hierárquico (GridSearch) |
| ------------------ | ------------------: | -------------------: | ----------------------: | -----------------------: |
| Black.people       |            0.033333 |             0.189003 |                0.153571 |                 0.185799 |
| Body               |            0.733289 |             0.767723 |                0.734328 |                 0.729750 |
| Fat.people         |            0.690047 |             0.764617 |                0.702403 |                 0.727286 |
| Fat.women          |            0.696795 |             0.766263 |                0.706346 |                 0.723673 |
| Feminists          |            0.145238 |             0.216007 |                0.156602 |                 0.200413 |
| Gays               |            0.132143 |             0.127879 |                0.139286 |                 0.100716 |
| Hate.speech        |            0.541421 |             0.541421 |                0.563487 |                 0.563487 |
| Homophobia         |            0.444841 |             0.371409 |                0.505877 |                 0.308055 |
| Homossexuals       |            0.478694 |             0.366807 |                0.501555 |                 0.301356 |
| Ideology           |            0.073333 |             0.162695 |                0.085618 |                 0.151854 |
| Immigrants         |            0.066667 |             0.166667 |                0.066667 |                 0.066667 |
| Islamists          |            0.000000 |             0.106667 |                0.000000 |                 0.155556 |
| Left.wing.ideology |            0.000000 |             0.000000 |                0.000000 |                 0.000000 |
| Lesbians           |            0.518246 |             0.377535 |                0.551797 |                 0.304095 |
| Men                |            0.111667 |             0.200905 |                0.132222 |                 0.215201 |
| Migrants           |            0.222323 |             0.348830 |                0.197807 |                 0.380267 |
| Muslims            |            0.100000 |             0.050000 |                0.100000 |                 0.050000 |
| Origin             |            0.000000 |             0.000000 |                0.000000 |                 0.000000 |
| OtherLifestyle     |            0.000000 |             0.000000 |                0.000000 |                 0.066667 |
| Racism             |            0.093333 |             0.252206 |                0.163800 |                 0.259246 |
| Refugees           |            0.216919 |             0.375196 |                0.206032 |                 0.375196 |
| Religion           |            0.050000 |             0.178333 |                0.050000 |                 0.185476 |
| Sexism             |            0.578073 |             0.461772 |                0.594783 |                 0.453496 |
| Trans.women        |            0.133333 |             0.050000 |                0.116667 |                 0.050000 |
| Transexuals        |            0.000000 |             0.000000 |                0.000000 |                 0.000000 |
| Ugly.people        |            0.677110 |             0.763622 |                0.679860 |                 0.702461 |
| Ugly.women         |            0.657170 |             0.697669 |                0.633477 |                 0.697669 |
| Women              |            0.579154 |             0.442695 |                0.589971 |                 0.445623 |




