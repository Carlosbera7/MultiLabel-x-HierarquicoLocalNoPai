import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========================
# Dados dos dois modelos
# ========================

# 🔹 Resultados Multilabel
data_multilabel = {
'Hate.speech':0.556628,'Sexism':0.595034,'Body':0.753551,'Racism':0.086900,'Ideology':0.094545,'Homophobia':0.480566,
'Origin':0.000000,'Religion':0.100000,'OtherLifestyle':0.000000,'Fat.people':0.723779,'Left.wing.ideology':0.000000,
'Ugly.people':0.696248,'Black.people':0.095238,'Fat.women':0.719905,'Feminists':0.120238,'Gays':0.139286,
'Immigrants':0.133333,'Islamists':0.000000,'Lesbians':0.559725,'Men':0.072222,'Muslims':0.100000,'Refugees':0.193629,
'Trans.women':0.133333,'Women':0.587269,'Transexuals':0.000000,'Ugly.women':0.679299,'Migrants':0.197958,'Homossexuals':0.517167
}

# 🔹 Resultados Hierárquico
data_hier = {
'Fat.women':0.774685,'Body':0.769534,'Fat.people':0.767709,'Ugly.people':0.759126,'Ugly.women':0.729310,
'Sexism':0.451818,'Women':0.431698,'Migrants':0.386431,'Refugees':0.378329,'Hate.speech':0.356145,'Homophobia':0.337062,
'Lesbians':0.330604,'Homossexuals':0.328174,'Feminists':0.265580,'Racism':0.262749,'Men':0.239939,'Black.people':0.196057,
'Ideology':0.181721,'Religion':0.167143,'Immigrants':0.133333,'Muslims':0.106667,'Gays':0.101515,'Transexuals':0.100000,
'Trans.women':0.050000,'Left.wing.ideology':0.025000,'Islamists':0.000000,'OtherLifestyle':0.000000,'Origin':0.000000
}

# ========================
# Cria DataFrames e alinha labels
# ========================
df_multilabel = pd.DataFrame.from_dict(data_multilabel, orient='index', columns=['F1_multilabel'])
df_hier = pd.DataFrame.from_dict(data_hier, orient='index', columns=['F1_hierarquico'])

# Junta os dois resultados e preenche com 0 caso alguma label não exista em um modelo
df_compare = df_multilabel.join(df_hier, how='outer').fillna(0)
df_compare = df_compare.sort_values(by='F1_multilabel', ascending=False)

# ========================
# Gráfico comparativo
# ========================
labels = df_compare.index
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width/2, df_compare['F1_multilabel'], width, label='Multilabel', alpha=0.8)
bars2 = ax.bar(x + width/2, df_compare['F1_hierarquico'], width, label='Hierárquico', alpha=0.8)

ax.set_ylabel('F1 Médio')
ax.set_title('Comparação do F1 Médio por Label (Multilabel vs Hierárquico)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90)
ax.legend()

plt.tight_layout()
plt.show()
