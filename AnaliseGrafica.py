import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
url = "https://raw.githubusercontent.com/Flavicentin/CreditoML/main/solicitacoescredito.csv"
df = pd.read_csv(url)

# 1. Analisar a distribuição da Margem Bruta Acumulada para todos os casos
plt.figure(figsize=(10, 6))
sns.histplot(df['margemBrutaAcumulada'], kde=True, bins=30)
plt.title('Distribuição da Margem Bruta Acumulada para Todos os Casos', fontsize=16)
plt.xlabel('Margem Bruta Acumulada', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.grid(True)
plt.show()

# 2. Comparar a Margem Bruta Acumulada por todos os status
plt.figure(figsize=(10, 6))
sns.boxplot(x='status', y='margemBrutaAcumulada', data=df)
plt.title('Margem Bruta Acumulada por Status', fontsize=16)
plt.xlabel('Status', fontsize=12)
plt.ylabel('Margem Bruta Acumulada', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# 3. Verificar estatísticas descritivas da Margem Bruta Acumulada para todos os casos
estatisticas = df['margemBrutaAcumulada'].describe()
print("\nEstatísticas descritivas da Margem Bruta Acumulada para todos os casos:")
print(estatisticas)
