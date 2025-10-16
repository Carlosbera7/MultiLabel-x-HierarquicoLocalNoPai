import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

# =======================
# Configurações do modelo
# =======================
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# =======================
# Hierarquia
# =======================
hierarchy = {
    'Hate.speech': ['Sexism', 'Body', 'Racism', 'Ideology', 'Homophobia', 'Origin', 'Religion', 'Migrants', 'OtherLifestyle'],
    'Sexism': ['Women', 'Men', 'Transexuals'],
    'Body': ['Fat.people', 'Ugly.people'],
    'Racism': ['Black.people'],
    'Ideology': ['Left.wing.ideology', 'Feminists'],
    'Homophobia': ['Homossexuals'],
    'Homossexuals': ['Gays', 'Lesbians'],
    'Women': ['Trans.women', 'Fat.women', 'Ugly.women'],
    'Migrants': ['Immigrants', 'Refugees'],
    'Religion': ['Islamists', 'Muslims'],
    'Fat.people': [],
    'Ugly.people': [],
    'Black.people': [],
    'Left.wing.ideology': [],
    'Feminists': [],
    'Gays': [],
    'Lesbians': [],
    'Men': [],
    'Transexuals': [],
    'Trans.women': [],
    'Fat.women': [],
    'Ugly.women': [],
    'Immigrants': [],
    'Refugees': [],
    'Islamists': [],
    'Muslims': [],
    'Origin': [],
    'OtherLifestyle': []
}

# =======================
# Funções auxiliares
# =======================
def train_local_parent(X_train, y_train, hierarchy):
    models = {}
    for parent, children in hierarchy.items():
        if not children:
            continue
        mask = y_train[parent] == 1
        if mask.sum() == 0:
            continue
        X_parent = X_train[mask]
        y_parent = y_train.loc[mask, children]
        models[parent] = {}
        for child in children:
            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_parent, y_parent[child])
            models[parent][child] = model
    return models


def predict_hierarchical(models, X_test, hierarchy):
    y_pred = pd.DataFrame(0, index=X_test.index, columns=list(hierarchy.keys()))
    y_pred['Hate.speech'] = 1
    for parent, children in hierarchy.items():
        if not children:
            continue
        mask = y_pred[parent] == 1
        if mask.sum() == 0:
            continue
        for child in children:
            model = models.get(parent, {}).get(child)
            if model is not None:
                preds = (model.predict_proba(X_test.loc[mask])[:, 1] > 0.5).astype(int)
                y_pred.loc[mask, child] = preds
    return y_pred


# =======================
# Execução com F1 médio por label
# =======================
f1_acumulado = {}

for fold in range(1, 11):
    print(f"\n===== Fold {fold} =====")

    X_train = pd.read_csv(f"folds/X_train_cv_xgb_{fold}.csv")
    y_train = pd.read_csv(f"folds/y_train_cv_xgb_{fold}.csv")
    X_test = pd.read_csv(f"folds/X_test_cv_xgb_{fold}.csv")
    y_test = pd.read_csv(f"folds/y_test_cv_xgb_{fold}.csv")

    models = train_local_parent(X_train, y_train, hierarchy)
    y_pred = predict_hierarchical(models, X_test, hierarchy)

    for label in y_test.columns:
        f1 = f1_score(y_test[label], y_pred[label], zero_division=0)
        if label not in f1_acumulado:
            f1_acumulado[label] = []
        f1_acumulado[label].append(f1)

# F1 médio por label (média dos folds)
f1_medio = {label: np.mean(scores) for label, scores in f1_acumulado.items()}
f1_df = pd.DataFrame.from_dict(f1_medio, orient='index', columns=['F1_medio']).sort_values(by='F1_medio', ascending=False)

print("\n===== F1 MÉDIO POR LABEL =====")
print(f1_df)

# opcional: salvar em CSV
f1_df.to_csv("f1_medio_por_label.csv", index=True)
