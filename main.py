import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
url = "https://raw.githubusercontent.com/Flavicentin/CreditoML/main/solicitacoescredito.csv"
df = pd.read_csv(url)

# Remover linhas com status 'AguardandoAprovacao'
df = df[df['status'] != 'AguardandoAprovacao']

# Mapeamento de status, definicaoRisco e intervaloFundacao
status_mapping = {
    'ReprovadoComite': 1,
    'EmAnaliseDocumentacao': 2,
    'DocumentacaoReprovada': 3,
    'AprovadoComite': 4,
    'ReprovadoAnalista': 5,
    'AprovadoAnalista': 6
}

definicao_risco_mapping = {
    'De 0 a 10 % - Muito Baixo': 0,
    'De 11 a 30 % - Baixo': 1,
    'De 31 a 50 % - Médio': 2,
    'De 51 a 80 % - Alto': 3
}

intervalo_fundacao_mapping = {
    'De 0 a 5 anos': 0,
    'De 11 a 16 anos': 1,
    'De 6 a 10 anos': 2,
    'Acima de 17 anos': 3
}

# Preencher valores nulos na coluna 'status' antes de mapear
df['status'] = df['status'].fillna('ReprovadoAnalista')

# Mapear as colunas categóricas para inteiros
df['status'] = df['status'].map(status_mapping).astype(int)
df['definicaoRisco'] = df['definicaoRisco'].map(definicao_risco_mapping).astype(int)

# Preencher valores nulos na coluna 'intervaloFundacao' com um valor padrão antes de mapear
df['intervaloFundacao'] = df['intervaloFundacao'].fillna('De 0 a 5 anos')
df['intervaloFundacao'] = df['intervaloFundacao'].map(intervalo_fundacao_mapping).astype(int)

# Garantir que os valores em 'empresa_MeEppMei' e 'restricoes' são strings
df['empresa_MeEppMei'] = df['empresa_MeEppMei'].astype(str)
df['restricoes'] = df['restricoes'].astype(str)

# Tratar 'empresa_MeEppMei' e 'restricoes', mapeando os valores:
# False -> 0, True -> 1, Nulo (ou 'nan') -> 2
df['empresa_MeEppMei'] = df['empresa_MeEppMei'].map({'False': 0, 'True': 1}).fillna(2).astype(int)
df['restricoes'] = df['restricoes'].map({'False': 0, 'True': 1}).fillna(2).astype(int)

# Tratar valores faltantes em outras colunas
df['valorAprovado'] = df['valorAprovado'].fillna(0)

# Definir um valor especial para dados ausentes
VALOR_AUSENTE = -9999

# Função para tratar colunas de data
def tratar_coluna_data(coluna):
    df[coluna] = pd.to_datetime(df[coluna], errors='coerce')  # Converter para datetime
    df[coluna] = df[coluna].fillna(pd.Timestamp("1900-01-01"))  # Preencher valores nulos
    df[coluna] = df[coluna].dt.strftime('%d/%m/%Y')  # Formatar para o formato desejado

# Aplicar o tratamento de datas para as colunas relevantes
colunas_data = ['primeiraCompra', 'dataAprovadoNivelAnalista', 'dataAprovadoEmComite']
for coluna in colunas_data:
    tratar_coluna_data(coluna)

# Tratamento específico para a coluna 'periodoBalanco'
def tratar_data_periodo_balanco(coluna):
    df[coluna] = pd.to_datetime(df[coluna], errors='coerce')
    df[coluna] = df[coluna].apply(lambda x: pd.NaT if x and (x.year < 1900 or x.year > 2024) else x)
    df[coluna] = df[coluna].fillna(pd.Timestamp("1900-01-01"))
    df[coluna] = df[coluna].dt.strftime('%d/%m/%Y')

# Aplicar o tratamento de data para a coluna 'periodoBalanco'
tratar_data_periodo_balanco('periodoBalanco')

# Preencher valores nulos nas colunas associadas com um identificador especial
colunas_associadas = ['ativoCirculante', 'passivoCirculante', 'totalAtivo',
                      'totalPatrimonioLiquido', 'endividamento', 'duplicatasAReceber', 'estoque']
for coluna in colunas_associadas:
    df[coluna] = df[coluna].fillna(VALOR_AUSENTE)

# Preencher os valores nulos em outras colunas com -9999
colunas_para_preencher = [
    'percentualProtestos', 'faturamentoBruto', 'margemBruta',
    'periodoDemonstrativoEmMeses', 'custos', 'anoFundacao',
    'capitalSocial', 'limiteEmpresaAnaliseCredito'
]
for coluna in colunas_para_preencher:
    df[coluna] = df[coluna].fillna(VALOR_AUSENTE)

# Criar coluna 'jaCliente' (1 para quem já teve uma compra, 0 para quem não teve)
df['jaCliente'] = df['primeiraCompra'].apply(lambda x: 1 if x != '01/01/1900' else 0)

# Calcular 'anosRelacao'
df['anosRelacao'] = df['primeiraCompra'].apply(
    lambda x: pd.Timestamp.now().year - pd.to_datetime(x, format='%d/%m/%Y').year if x != '01/01/1900' else 0)

# Criar coluna 'mobCliente' (classificação do relacionamento)
def classificar_relacionamento(anos):
    if 1 <= anos <= 5:
        return 1  # Relacionamento baixo
    elif 6 <= anos <= 10:
        return 2  # Relacionamento médio
    elif anos > 10:
        return 3  # Relacionamento alto
    else:
        return 0

df['mobCliente'] = df['anosRelacao'].apply(classificar_relacionamento)

# Criar a coluna 'passagemCliente' (relação de aprovações)
def calcular_passagem(cnpj):
    total_propostas = len(df[df['cnpjSemTraco'] == cnpj])
    propostas_aprovadas = len(df[(df['cnpjSemTraco'] == cnpj) & (df['status'].isin([4, 6]))])
    if total_propostas == 0:
        return 0.0
    return round(propostas_aprovadas / total_propostas, 1)

df['passagemCliente'] = df['cnpjSemTraco'].apply(calcular_passagem)

# Verificar valores faltantes nas colunas
missing_values = df.isna().sum()
missing_columns = missing_values[missing_values > 0]
print("Colunas com valores faltantes após o tratamento:")
print(missing_columns)

# Salvar o arquivo CSV transformado
df.to_csv('solicitacoescredito_transformado_final.csv', index=False)
print("Arquivo CSV final transformado com sucesso!")
