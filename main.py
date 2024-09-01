import pandas as pd

# Use uma das duas opções abaixo para o caminho do arquivo
url = r"C:\Users\flavi\PycharmProjects\MachineLearning\solicitacoescredito.csv"

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

