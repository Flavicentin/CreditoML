import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
url = "https://raw.githubusercontent.com/Flavicentin/CreditoML/main/solicitacoescredito.csv"
df = pd.read_csv(url)

# Preencher valores nulos antes de aplicar os mapeamentos
df['status'] = df['status'].fillna('ReprovadoAnalista')
df['definicaoRisco'] = df['definicaoRisco'].fillna('De 0 a 10 % - Muito Baixo')
df['intervaloFundacao'] = df['intervaloFundacao'].fillna('De 0 a 5 anos')

# Inicializar o LabelEncoder para cada coluna
label_encoder_status = LabelEncoder()
label_encoder_definicao_risco = LabelEncoder()
label_encoder_intervalo_fundacao = LabelEncoder()

# Adicionar colunas para manter os valores originais e colocar as novas colunas no final
df['statusOriginal'] = df['status']
df['classificacaoStatus'] = label_encoder_status.fit_transform(df['status'])
df['definicaoRiscoOriginal'] = df['definicaoRisco']
df['classificacaoDefinicaoRisco'] = label_encoder_definicao_risco.fit_transform(df['definicaoRisco'])
df['intervaloFundacaoOriginal'] = df['intervaloFundacao']
df['classificacaoIntervaloFundacao'] = label_encoder_intervalo_fundacao.fit_transform(df['intervaloFundacao'])

# Garantir que os valores em 'empresa_MeEppMei' e 'restricoes' são strings e mapear
df['empresa_MeEppMei'] = df['empresa_MeEppMei'].astype(str)
df['restricoes'] = df['restricoes'].astype(str)
df['empresa_MeEppMei'] = df['empresa_MeEppMei'].map({'False': 0, 'True': 1}).fillna(2).astype(int)
df['restricoes'] = df['restricoes'].map({'False': 0, 'True': 1}).fillna(2).astype(int)

# Tratar valores faltantes em outras colunas
df['valorAprovado'] = df['valorAprovado'].fillna(0)

# Definir um valor especial para dados ausentes
VALOR_AUSENTE = -9999

# Função para tratar colunas de data
def tratar_coluna_data(coluna):
    df[coluna] = pd.to_datetime(df[coluna], errors='coerce')
    df[coluna] = df[coluna].fillna(pd.Timestamp("1900-01-01"))
    df[coluna] = df[coluna].dt.strftime('%d/%m/%Y')

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
    'capitalSocial', 'limiteEmpresaAnaliseCredito', 'maiorAtraso'
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
    propostas_aprovadas = len(df[(df['cnpjSemTraco'] == cnpj) & (df['status'].isin(['AprovadoAnalista', 'AprovadoEmComite']))])
    if total_propostas == 0:
        return 0.0
    return round(propostas_aprovadas / total_propostas, 1)

df['passagemCliente'] = df['cnpjSemTraco'].apply(calcular_passagem)

# -------------------------
# Tratamento de 'totalPatrimonioLiquido'

# Converter 'totalPatrimonioLiquido' para numérico e substituir -9999 por NaN
df['totalPatrimonioLiquido'] = pd.to_numeric(df['totalPatrimonioLiquido'], errors='coerce')
df['totalPatrimonioLiquido'] = df['totalPatrimonioLiquido'].replace(VALOR_AUSENTE, np.nan)

# Calcular os percentis necessários
percentis_patrimonio = np.percentile(df['totalPatrimonioLiquido'].dropna(), [0, 20, 40, 60, 80, 100])

# Definir a função de classificação
def classificar_patrimonio(valor):
    if pd.isnull(valor):
        return 'Dados Ausentes'
    elif valor < 0:
        return 'Negativo'
    elif 0 <= valor < percentis_patrimonio[1]:
        return 'Muito Baixo'
    elif percentis_patrimonio[1] <= valor < percentis_patrimonio[2]:
        return 'Baixo'
    elif percentis_patrimonio[2] <= valor < percentis_patrimonio[3]:
        return 'Médio'
    elif percentis_patrimonio[3] <= valor < percentis_patrimonio[4]:
        return 'Alto'
    else:
        return 'Muito Alto'

# Aplicar a classificação
df['classificacaoTotalPatrimonioLiquido'] = df['totalPatrimonioLiquido'].apply(classificar_patrimonio)

# Aplicar LabelEncoder em 'classificacaoTotalPatrimonioLiquido'
mapping_patrimonio = {
    'Dados Ausentes': 0,
    'Negativo': 1,
    'Muito Baixo': 2,
    'Baixo': 3,
    'Médio': 4,
    'Alto': 5,
    'Muito Alto': 6
}
df['classificacaoTotalPatrimonioLiquidoEncoded'] = df['classificacaoTotalPatrimonioLiquido'].map(mapping_patrimonio)

# -------------------------
# Tratamento de 'endividamento'

# Garantindo que 'endividamento' seja numérico
df['endividamento'] = pd.to_numeric(df['endividamento'], errors='coerce')

# Identificando valores -9999 (dados ausentes)
df['endividamento_ausente'] = df['endividamento'] == VALOR_AUSENTE

# Substituindo valores -9999 por NaN
df.loc[df['endividamento_ausente'], 'endividamento'] = np.nan

# Dados com endividamento ausente
df.loc[df['endividamento'].isna(), 'endividamento_categoria'] = 'Dados Ausentes'

# Dados com endividamento zero
df.loc[df['endividamento'] == 0, 'endividamento_categoria'] = 'Sem Endividamento'

# Dados com endividamento positivo
df_positivo_endividamento = df[df['endividamento'] > 0].copy()

# Calculando percentis para valores positivos
percentis_endividamento = np.percentile(df_positivo_endividamento['endividamento'], [20, 40, 60, 80, 100])

# Definindo bins e labels para valores positivos
bins_endividamento = [0] + list(percentis_endividamento)
labels_endividamento = ['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Muito Alto']
bins_endividamento = sorted(set(bins_endividamento))

# Aplicando a classificação aos valores positivos
df.loc[df['endividamento'] > 0, 'endividamento_categoria'] = pd.cut(
    df.loc[df['endividamento'] > 0, 'endividamento'],
    bins=bins_endividamento, labels=labels_endividamento, include_lowest=True
)

# Converter para string para evitar problemas com categorias
df['endividamento_categoria'] = df['endividamento_categoria'].astype(str)

# Mapeando 'endividamento_categoria' para números conforme especificado
mapping_endividamento = {
    'Dados Ausentes': 0,
    'Muito Alto': 1,
    'Alto': 2,
    'Médio': 3,
    'Baixo': 4,
    'Muito Baixo': 5,
    'Sem Endividamento': 6
}

df['classificacaoEndividamento'] = df['endividamento_categoria'].map(mapping_endividamento)

# Removendo colunas auxiliares
df.drop(['endividamento_ausente', 'endividamento_categoria'], axis=1, inplace=True)

# -------------------------
# Tratamento de 'capitalSocial'

# Converter 'capitalSocial' para numérico e substituir -9999 por NaN
df['capitalSocial'] = pd.to_numeric(df['capitalSocial'], errors='coerce')
df['capitalSocial'] = df['capitalSocial'].replace(VALOR_AUSENTE, np.nan)

# Initialize 'capitalSocial_categoria' with dtype=object to avoid FutureWarning
df['capitalSocial_categoria'] = pd.Series(dtype='object')

# Separar dados positivos
df_positivo_capital = df[df['capitalSocial'] > 0]

# Calculando percentis para valores positivos
percentis_capital_social = np.percentile(df_positivo_capital['capitalSocial'], [20, 40, 60, 80, 100])

# Definindo bins e labels
bins_capital = [0] + list(percentis_capital_social)
labels_capital = ['Muito Baixa', 'Baixa', 'Média', 'Alto', 'Muito Alto']
bins_capital = sorted(set(bins_capital))

# Aplicando a classificação aos valores positivos
df.loc[df['capitalSocial'] > 0, 'capitalSocial_categoria'] = pd.cut(
    df.loc[df['capitalSocial'] > 0, 'capitalSocial'],
    bins=bins_capital, labels=labels_capital, include_lowest=True
)

# Converter para string para evitar problemas com categorias
df['capitalSocial_categoria'] = df['capitalSocial_categoria'].astype(str)

# Atribuir categorias aos outros grupos
df.loc[df['capitalSocial'] == 0, 'capitalSocial_categoria'] = 'Sem Capital'
df.loc[df['capitalSocial'] < 0, 'capitalSocial_categoria'] = 'Negativo'
df.loc[df['capitalSocial'].isna(), 'capitalSocial_categoria'] = 'Dados Ausentes'

# Mapeando 'capitalSocial_categoria' para números conforme especificado
mapping_capital_social = {
    'Muito Baixa': 0,
    'Baixa': 1,
    'Média': 2,
    'Alto': 3,
    'Muito Alto': 4,
    'Sem Capital': 5,
    'Negativo': 6,
    'Dados Ausentes': 7
}

df['classificacaoCapitalSocial'] = df['capitalSocial_categoria'].map(mapping_capital_social)

# Remover colunas auxiliares
df.drop(['capitalSocial_categoria'], axis=1, inplace=True)

# -------------------------
# Tratamento de 'faturamentoBruto'

# Converter 'faturamentoBruto' para numérico e substituir -9999 por NaN
df['faturamentoBruto'] = pd.to_numeric(df['faturamentoBruto'], errors='coerce')
df['faturamentoBruto'] = df['faturamentoBruto'].replace(VALOR_AUSENTE, np.nan)

# Initialize 'faturamentoBruto_categoria' com dtype=object
df['faturamentoBruto_categoria'] = pd.Series(dtype='object')

# Separar dados positivos
df_positivo_faturamento = df[df['faturamentoBruto'] > 0]

# Calculando percentis para valores positivos
percentis_faturamento_bruto = np.percentile(df_positivo_faturamento['faturamentoBruto'], [20, 40, 60, 80, 100])

# Definindo bins e labels
bins_faturamento = [0] + list(percentis_faturamento_bruto)
labels_faturamento = ['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Muito Alto']
bins_faturamento = sorted(set(bins_faturamento))

# Aplicando a classificação aos valores positivos
df.loc[df['faturamentoBruto'] > 0, 'faturamentoBruto_categoria'] = pd.cut(
    df.loc[df['faturamentoBruto'] > 0, 'faturamentoBruto'],
    bins=bins_faturamento, labels=labels_faturamento, include_lowest=True
)

# Converter para string para evitar problemas com categorias
df['faturamentoBruto_categoria'] = df['faturamentoBruto_categoria'].astype(str)

# Atribuir categorias aos outros grupos
df.loc[df['faturamentoBruto'] == 0, 'faturamentoBruto_categoria'] = 'Sem Faturamento'
df.loc[df['faturamentoBruto'] < 0, 'faturamentoBruto_categoria'] = 'Negativo'
df.loc[df['faturamentoBruto'].isna(), 'faturamentoBruto_categoria'] = 'Dados Ausentes'

# Aplicar LabelEncoder em 'faturamentoBruto_categoria'
label_encoder_faturamento_bruto = LabelEncoder()
df['classificacaoFaturamentoBruto'] = label_encoder_faturamento_bruto.fit_transform(df['faturamentoBruto_categoria'])

# Remover colunas auxiliares
df.drop(['faturamentoBruto_categoria'], axis=1, inplace=True)

# -------------------------
# Tratamento de 'margemBruta'

# Verificar se 'margemBrutaAcumulada' existe; caso contrário, usar 'margemBruta'
if 'margemBrutaAcumulada' in df.columns:
    margem_bruta_col = 'margemBrutaAcumulada'
elif 'margemBruta' in df.columns:
    margem_bruta_col = 'margemBruta'
else:
    raise ValueError("As colunas 'margemBrutaAcumulada' e 'margemBruta' não estão presentes no conjunto de dados.")

# Converter para numérico e tratar placeholders
df[margem_bruta_col] = pd.to_numeric(df[margem_bruta_col], errors='coerce')
df[margem_bruta_col] = df[margem_bruta_col].replace(VALOR_AUSENTE, np.nan)

# Initialize 'margemBruta_categoria' com dtype=object
df['margemBruta_categoria'] = pd.Series(dtype='object')

# Separar dados positivos
df_positivo_margem = df[df[margem_bruta_col] > 0]

# Calculando percentis para valores positivos
percentis_margem_bruta = np.percentile(df_positivo_margem[margem_bruta_col], [20, 40, 60, 80, 100])

# Definindo bins e labels
bins_margem = [0] + list(percentis_margem_bruta)
labels_margem = ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']
bins_margem = sorted(set(bins_margem))

# Aplicando a classificação aos valores positivos
df.loc[df[margem_bruta_col] > 0, 'margemBruta_categoria'] = pd.cut(
    df.loc[df[margem_bruta_col] > 0, margem_bruta_col],
    bins=bins_margem, labels=labels_margem, include_lowest=True
)

# Converter para string para evitar problemas com categorias
df['margemBruta_categoria'] = df['margemBruta_categoria'].astype(str)

# Atribuir categorias aos outros grupos
df.loc[df[margem_bruta_col] == 0, 'margemBruta_categoria'] = 'Sem Margem Bruta'
df.loc[df[margem_bruta_col] < 0, 'margemBruta_categoria'] = 'Negativa'
df.loc[df[margem_bruta_col].isna(), 'margemBruta_categoria'] = 'Dados Ausentes'

# Aplicar LabelEncoder em 'margemBruta_categoria'
label_encoder_margem_bruta = LabelEncoder()
df['classificacaoMargemBruta'] = label_encoder_margem_bruta.fit_transform(df['margemBruta_categoria'])

# Remover colunas auxiliares
df.drop(['margemBruta_categoria'], axis=1, inplace=True)

# -------------------------
# Tratamento de 'maiorAtraso'

# Verificar se 'maiorAtraso' existe
if 'maiorAtraso' in df.columns:
    # Converter para numérico e tratar placeholders
    df['maiorAtraso'] = pd.to_numeric(df['maiorAtraso'], errors='coerce')
    df['maiorAtraso'] = df['maiorAtraso'].replace(VALOR_AUSENTE, np.nan)

    # Initialize 'maiorAtraso_categoria' com dtype=object
    df['maiorAtraso_categoria'] = pd.Series(dtype='object')

    # Separar dados positivos
    df_positivo_atraso = df[df['maiorAtraso'] > 0]

    # Calculando percentis para valores positivos
    percentis_maior_atraso = np.percentile(df_positivo_atraso['maiorAtraso'], [20, 40, 60, 80, 100])

    # Definir bins e labels
    bins_atraso = [0] + list(percentis_maior_atraso)
    labels_atraso = ['Baixo', 'Moderado', 'Alto', 'Muito Alto', 'Extremo']
    bins_atraso = sorted(set(bins_atraso))

    # Aplicação da classificação aos valores positivos
    df.loc[df['maiorAtraso'] > 0, 'maiorAtraso_categoria'] = pd.cut(
        df.loc[df['maiorAtraso'] > 0, 'maiorAtraso'],
        bins=bins_atraso, labels=labels_atraso, include_lowest=True
    )

    # Converter para string para evitar problemas com categorias
    df['maiorAtraso_categoria'] = df['maiorAtraso_categoria'].astype(str)

    # Atribuir categorias aos outros grupos
    df.loc[df['maiorAtraso'] == 0, 'maiorAtraso_categoria'] = 'Sem Atraso'
    df.loc[df['maiorAtraso'] < 0, 'maiorAtraso_categoria'] = 'Valor Negativo'
    df.loc[df['maiorAtraso'].isna(), 'maiorAtraso_categoria'] = 'Dados Ausentes'

    # Mapeamento das categorias para números conforme especificado
    mapping_maior_atraso = {
        'Alto': 0,
        'Baixo': 1,
        'Extremo': 2,
        'Moderado': 3,
        'Muito Alto': 4,
        'Sem Atraso': 5,
        'Valor Negativo': 6,
        'Dados Ausentes': 7
    }

    df['classificacaoMaiorAtraso'] = df['maiorAtraso_categoria'].map(mapping_maior_atraso)

    # Remover colunas auxiliares
    df.drop(['maiorAtraso_categoria'], axis=1, inplace=True)
else:
    raise ValueError("A coluna 'maiorAtraso' não está presente no conjunto de dados.")

# -------------------------
# Arredondar 'scorePontualidade' para uma casa decimal
df['scorePontualidade'] = df['scorePontualidade'].astype(float).round(1)

# Atualizar a lista de colunas para inserir
colunas_para_inserir = [
    'classificacaoMargemBruta',
    'classificacaoMaiorAtraso',
    'classificacaoTotalPatrimonioLiquido', 'classificacaoTotalPatrimonioLiquidoEncoded',
    'classificacaoEndividamento',
    'classificacaoFaturamentoBruto',
    'classificacaoCapitalSocial'
]

# Rearranjar as colunas para mover 'colunas_para_inserir' para o início
df = df.reindex(columns=colunas_para_inserir + [col for col in df.columns if col not in colunas_para_inserir])

# Visualizar as primeiras linhas para verificar
print(df.head())

# Salvar o arquivo transformado
df.to_csv('solicitacoescredito_transformado_final5.csv', index=False)
print("Arquivo CSV final transformado com sucesso!")
