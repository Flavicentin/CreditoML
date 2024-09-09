import pandas as pd
import matplotlib.pyplot as plt

# Carregar o dataset
df_credito = pd.read_csv("https://raw.githubusercontent.com/Flavicentin/CreditoML/main/solicitacoescredito.csv")

# Filtrar apenas os casos aprovados
status_aprovado = ['AprovadoAnalista', 'AprovadoComite']
df_aprovados = df_credito[df_credito['status'].isin(status_aprovado)]

# 1. Média
media_valor_solicitado = df_aprovados['valorSolicitado'].mean()
media_valor_aprovado = df_aprovados['valorAprovado'].mean()
print(f"Média dos Valores Solicitados (Aprovados): {media_valor_solicitado:.2f}")
print(f"Média dos Valores Aprovados: {media_valor_aprovado:.2f}")

# 2. Mediana
mediana_valor_solicitado = df_aprovados['valorSolicitado'].median()
mediana_valor_aprovado = df_aprovados['valorAprovado'].median()
print(f"Mediana dos Valores Solicitados (Aprovados): {mediana_valor_solicitado:.2f}")
print(f"Mediana dos Valores Aprovados: {mediana_valor_aprovado:.2f}")

# 3. Moda
moda_valor_solicitado = df_aprovados['valorSolicitado'].mode()[0]
moda_valor_aprovado = df_aprovados['valorAprovado'].mode()[0]
print(f"Moda dos Valores Solicitados (Aprovados): {moda_valor_solicitado:.2f}")
print(f"Moda dos Valores Aprovados: {moda_valor_aprovado:.2f}")

# 4. Quartis (Q1 e Q3)
Q1_valor_solicitado = df_aprovados['valorSolicitado'].quantile(0.25)
Q3_valor_solicitado = df_aprovados['valorSolicitado'].quantile(0.75)
Q1_valor_aprovado = df_aprovados['valorAprovado'].quantile(0.25)
Q3_valor_aprovado = df_aprovados['valorAprovado'].quantile(0.75)
print(f"Quartil 1 dos Valores Solicitados (Aprovados): {Q1_valor_solicitado:.2f}")
print(f"Quartil 3 dos Valores Solicitados (Aprovados): {Q3_valor_solicitado:.2f}")
print(f"Quartil 1 dos Valores Aprovados: {Q1_valor_aprovado:.2f}")
print(f"Quartil 3 dos Valores Aprovados: {Q3_valor_aprovado:.2f}")

# 5. Desvio Padrão
desvio_padrao_solicitado = df_aprovados['valorSolicitado'].std()
desvio_padrao_aprovado = df_aprovados['valorAprovado'].std()
print(f"Desvio Padrão dos Valores Solicitados (Aprovados): {desvio_padrao_solicitado:.2f}")
print(f"Desvio Padrão dos Valores Aprovados: {desvio_padrao_aprovado:.2f}")

# 6. Intervalo Interquartil (IQR)
iqr_solicitado = Q3_valor_solicitado - Q1_valor_solicitado
iqr_aprovado = Q3_valor_aprovado - Q1_valor_aprovado
print(f"Intervalo Interquartil dos Valores Solicitados (Aprovados): {iqr_solicitado:.2f}")
print(f"Intervalo Interquartil dos Valores Aprovados: {iqr_aprovado:.2f}")

# 7. Boxplot para visualizar os valores solicitados e aprovados (Aprovados)
plt.figure(figsize=(8,6))
df_aprovados[['valorSolicitado', 'valorAprovado']].boxplot()
plt.title('Boxplot dos Valores Solicitados e Aprovados (Casos Aprovados)', fontsize=16)
plt.ylabel('Valor', fontsize=12)
plt.grid(True)
plt.show()

# 8. Histograma para visualizar a distribuição dos valores solicitados (Aprovados)
plt.figure(figsize=(10,6))
df_aprovados['valorSolicitado'].hist(bins=30, color='skyblue')
plt.title('Histograma dos Valores Solicitados (Aprovados)', fontsize=16)
plt.xlabel('Valor Solicitado', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.grid(True)
plt.show()

# 9. Histograma para visualizar a distribuição dos valores aprovados (Aprovados)
plt.figure(figsize=(10,6))
df_aprovados['valorAprovado'].hist(bins=30, color='lightgreen')
plt.title('Histograma dos Valores Aprovados (Aprovados)', fontsize=16)
plt.xlabel('Valor Aprovado', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.grid(True)
plt.show()

# 10. Scatter plot da relação entre valor solicitado e valor aprovado (Aprovados)
plt.figure(figsize=(10,6))
plt.scatter(df_aprovados['valorSolicitado'], df_aprovados['valorAprovado'], alpha=0.5, color='purple')
plt.title('Relação entre Valor Solicitado e Valor Aprovado (Aprovados)', fontsize=16)
plt.xlabel('Valor Solicitado', fontsize=12)
plt.ylabel('Valor Aprovado', fontsize=12)
plt.grid(True)
plt.show()
