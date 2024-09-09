import pandas as pd

# URL do dataset
url = "https://raw.githubusercontent.com/Flavicentin/CreditoML/main/solicitacoescredito.csv"

# Lê e carrega o arquivo para a memória
df = pd.read_csv(url)

# Quantos dados existem nesse dataset?
num_dados = df.shape[0]
print(f"Total de dados: {num_dados}")

# Qual a quantidade de atributos?
num_atributos = df.shape[1]
print(f"Quantidade de atributos: {num_atributos}")

# Existe valores faltantes?
valores_faltantes = df.isnull().sum().sum()
print(f"Total de valores faltantes: {valores_faltantes}")

# De que tipo são os dados (dtype)?
tipos_dados = df.dtypes
print(f"Tipos de dados:\n{tipos_dados}")

# Quantidade de linhas sem valores faltantes
linhas_completas = df.dropna().shape[0]
print(f"Quantidade de linhas completas: {linhas_completas}")

# Verifica quais colunas têm valores ausentes e quantos sãoa
valores_faltantes_por_coluna = df.isnull().sum()

# Filtra para mostrar apenas as colunas que têm valores ausentes
colunas_com_faltantes = valores_faltantes_por_coluna[valores_faltantes_por_coluna > 0]

print("Colunas com valores ausentes e a quantidade de valores faltantes em cada uma:")
print(colunas_com_faltantes)

# Verificar quais colunas estão totalmente preenchidas
colunas_totalmente_preenchidas = df.columns[df.isnull().sum() == 0]
print("Colunas com todos os dados preenchidos:")
print(colunas_totalmente_preenchidas)

# Verificar quais colunas possuem dados faltantes
colunas_com_faltantes = df.columns[df.isnull().sum() > 0]
print("\nColunas com dados faltantes:")
print(colunas_com_faltantes)