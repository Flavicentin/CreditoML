import pandas as pd
import numpy as np

# Carregar o dataset
url = "https://raw.githubusercontent.com/Flavicentin/CreditoML/main/solicitacoescredito.csv"
df = pd.read_csv(url)

# Converter colunas de data para datetime
df['dataAprovadoEmComite'] = pd.to_datetime(df['dataAprovadoEmComite'], errors='coerce')
df['dataAprovadoNivelAnalista'] = pd.to_datetime(df['dataAprovadoNivelAnalista'], errors='coerce')

# Definir funções auxiliares para agregação
def most_frequent(x):
    return x.mode().iloc[0] if not x.mode().empty else np.nan

def latest_date(x):
    return x.max()

def count_status(x):
    return x.value_counts().to_dict()

def safe_max(x):
    """Aplica a função max apenas em colunas numéricas."""
    try:
        return x.astype(float).max()
    except:
        return np.nan

# Definir as colunas de agregação
aggregation_functions = {
    'razaoSocial': 'first',
    'nomeFantasia': 'first',
    'maiorAtraso': safe_max,  # Use safe_max para evitar erro com tipos mistos
    'margemBrutaAcumulada': 'mean',
    'percentualProtestos': 'mean',
    'primeiraCompra': 'first',  # Exemplo para coluna categórica, ajustar conforme necessário
    'prazoMedioRecebimentoVendas': 'mean',
    'titulosEmAberto': 'sum',
    'valorSolicitado': 'sum',
    'status': count_status,  # Será transformado posteriormente
    'definicaoRisco': most_frequent,
    'diferencaPercentualRisco': 'mean',
    'percentualRisco': 'mean',
    'dashboardCorrelacao': 'first',
    'valorAprovado': 'sum',
    'dataAprovadoEmComite': latest_date,
    'periodoBalanco': 'first',
    'ativoCirculante': 'mean',
    'passivoCirculante': 'mean',
    'totalAtivo': 'mean',
    'totalPatrimonioLiquido': 'mean',
    'endividamento': 'mean',
    'duplicatasAReceber': 'mean',
    'estoque': 'mean',
    'faturamentoBruto': 'mean',
    'margemBruta': 'mean',
    'periodoDemonstrativoEmMeses': 'mean',
    'custos': 'mean',
    'anoFundacao': 'min',
    'intervaloFundacao': 'first',
    'capitalSocial': 'mean',
    'restricoes': 'sum',
    'empresa_MeEppMei': most_frequent,
    'scorePontualidade': 'mean',
    'limiteEmpresaAnaliseCredito': 'mean',
    'dataAprovadoNivelAnalista': latest_date
}

# Agregar os dados por 'cnpjSemTraco'
df_aggregated = df.groupby('cnpjSemTraco').agg(aggregation_functions).reset_index()

# Tratar a coluna 'status' que agora é um dicionário
# Podemos transformar as contagens em múltiplas colunas ou mantê-las como estão
# Aqui, criaremos colunas para cada status com a contagem correspondente
status_df = df.groupby('cnpjSemTraco')['status'].apply(count_status).apply(pd.Series).fillna(0)

# Renomear as colunas para indicar que são contagens de status
status_df = status_df.rename(columns=lambda x: f'status_{x}')

# Concatenar com o dataframe agregado
df_aggregated = pd.concat([df_aggregated, status_df], axis=1)

# Exibir as primeiras linhas do dataframe agregado
print(df_aggregated.head())

# Opcional: Salvar o dataframe agregado em um novo arquivo CSV
df_aggregated.to_csv('solicitacoescredito_agregado.csv', index=False)
print("Arquivo 'solicitacoescredito_agregado.csv' gerado com sucesso.")
