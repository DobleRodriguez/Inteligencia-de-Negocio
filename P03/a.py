import pandas as pd

df = pd.read_csv('mis_resultados_rf.csv')
df['Precio_cat'] = df['Precio_cat'].str.extract(r'(\d)')
df.to_csv('mis_resultados_rf.csv')
