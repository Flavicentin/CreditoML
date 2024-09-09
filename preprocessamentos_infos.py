import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Carregar os dados
url = "https://raw.githubusercontent.com/Flavicentin/CreditoML/main/solicitacoescredito.csv"
df = pd.read_csv(url)

# Ordenar o dataframe por CNPJ e pela data da solicitação (exemplo com 'dataAprovadoNivelAnalista')
df.sort_values(by=['cnpjSemTraco', 'dataAprovadoNivelAnalista'], inplace=True)

# Colunas para preencher a partir de solicitações anteriores aprovadas
colunas_para_preencher = [
    'percentualProtestos', 'valorAprovado', 'ativoCirculante',
    'passivoCirculante', 'totalAtivo', 'totalPatrimonioLiquido',
    'endividamento', 'duplicatasAReceber', 'estoque', 'faturamentoBruto',
    'margemBruta', 'capitalSocial'
]

# Criar uma máscara que verifica se o status é "aprovado"
status_aprovado = ['AprovadoAnalista', 'AprovadoComite']

# Preencher os valores faltantes com base nas solicitações aprovadas anteriores dentro de cada grupo de CNPJ
for coluna in colunas_para_preencher:
    df[coluna] = df.groupby('cnpjSemTraco')[coluna].transform(lambda x: x.ffill())

# Para colunas específicas com poucos valores faltantes, como 'primeiraCompra', usar o valor mais frequente (moda)
df['primeiraCompra'] = df['primeiraCompra'].fillna(df['primeiraCompra'].mode()[0])

# Excluir colunas com muitos valores faltantes que não são relevantes
colunas_para_excluir = ['dataAprovadoEmComite', 'periodoBalanco']
df = df.drop(columns=colunas_para_excluir)

# 1. Tratamento de valores faltantes com SimpleImputer
imputer = SimpleImputer(strategy='median')
df[colunas_para_preencher] = imputer.fit_transform(df[colunas_para_preencher])

# 2. Normalização dos dados numéricos
cols_numericas = ['ativoCirculante', 'passivoCirculante', 'totalAtivo',
                  'totalPatrimonioLiquido', 'endividamento', 'faturamentoBruto']

scaler_minmax = MinMaxScaler()
df[cols_numericas] = scaler_minmax.fit_transform(df[cols_numericas])

onehotencoder = OneHotEncoder()
empresa_encoded = onehotencoder.fit_transform(df[['empresa_MeEppMei']]).toarray()
df = pd.concat([df, pd.DataFrame(empresa_encoded, columns=onehotencoder.get_feature_names_out(['empresa_MeEppMei']))], axis=1)
df = df.drop(columns=['empresa_MeEppMei'])

# 4. Redução de dimensionalidade com PCA
pca = PCA(n_components=0.90)
dados_pca = pca.fit_transform(df[cols_numericas])

df_pca = pd.DataFrame(dados_pca, columns=[f"pca_{i}" for i in range(dados_pca.shape[1])])
df = pd.concat([df, df_pca], axis=1)

# Verificar a variância explicada por cada componente do PCA
print("Variância explicada por cada componente do PCA:", pca.explained_variance_ratio_)

# 5. Gerar o novo arquivo CSV com os dados pré-processados
df.to_csv('solicitacoescredito_preprocessado_otimizado.csv', index=False)
print("Arquivo 'solicitacoescredito_preprocessado_otimizado.csv' gerado com sucesso.")