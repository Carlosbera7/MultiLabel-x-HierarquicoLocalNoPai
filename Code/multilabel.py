import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

# parâmetros fixos do modelo
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",        
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# dicionário para armazenar F1 de cada label em cada fold
f1_by_label = []

for fold in range(1, 11):  # 10 folds
    print(f"\n===== Fold {fold} =====")
    
    # lê os arquivos do fold
    X_train = pd.read_csv(f'folds/X_train_cv_xgb_{fold}.csv')
    X_test = pd.read_csv(f'folds/X_test_cv_xgb_{fold}.csv')
    y_train = pd.read_csv(f'folds/y_train_cv_xgb_{fold}.csv')
    y_test = pd.read_csv(f'folds/y_test_cv_xgb_{fold}.csv')
    
    # cria o modelo base do XGBoost
    xgb_model = XGBClassifier(**XGB_PARAMS)
    
    # modelo multilabel
    model = MultiOutputClassifier(xgb_model, n_jobs=-1)
    
    # treinamento
    model.fit(X_train, y_train)
    
    # predição
    y_pred = model.predict(X_test)
    
    # calcula o F1-score para cada label individualmente
    f1_labels = f1_score(y_test, y_pred, average=None)
    f1_by_label.append(f1_labels)
    
    print(f"F1 médio do fold {fold}: {np.mean(f1_labels):.4f}")

# converte os resultados em DataFrame
f1_df = pd.DataFrame(f1_by_label, columns=y_train.columns)

# calcula média e desvio padrão por label
f1_mean = f1_df.mean().rename("F1_medio")
f1_std = f1_df.std().rename("F1_desvio")

# combina tudo em um único DataFrame
f1_summary = pd.concat([f1_mean, f1_std], axis=1)

print("\n==== F1 MÉDIO POR LABEL (10 folds) ====")
print(f1_summary)

# salva em CSV opcionalmente
f1_summary.to_csv("f1_por_label_xgb.csv", index=True)
print("\nResultados salvos em: f1_por_label_xgb.csv")
