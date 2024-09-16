import pandas as pd
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
    propostas_aprovadas = len(df[(df['cnpjSemTraco'] == cnpj) & (df['status'].isin(['AprovadoAnalista', 'AprovadoEmComite']))])
    if total_propostas == 0:
        return 0.0
    return round(propostas_aprovadas / total_propostas, 1)

df['passagemCliente'] = df['cnpjSemTraco'].apply(calcular_passagem)

# -------------------------
# Definir faixas para o totalPatrimonioLiquido
def classificar_patrimonio(valor):
    if valor == VALOR_AUSENTE:
        return 'Ausente'
    elif valor < 0:
        return 'Negativo'
    elif 0 <= valor < 100000:
        return 'Baixo'
    elif 100000 <= valor < 500000:
        return 'Médio'
    elif 500000 <= valor < 1000000:
        return 'Alto'
    else:
        return 'Muito Alto'

# Criar a nova coluna com as classificações para totalPatrimonioLiquido
df['classificacaoTotalPatrimonioLiquido'] = df['totalPatrimonioLiquido'].apply(classificar_patrimonio)

# Aplicar o LabelEncoder para converter as classificações em valores numéricos
label_encoder_patrimonio = LabelEncoder()
df['classificacaoTotalPatrimonioLiquidoEncoded'] = label_encoder_patrimonio.fit_transform(df['classificacaoTotalPatrimonioLiquido'])

# Adicionar a categorização e codificação de margemBruta
# Verificar se 'margemBrutaAcumulada' existe; se não, usar 'margemBruta'
if 'margemBrutaAcumulada' in df.columns:
    margem_bruta_col = 'margemBrutaAcumulada'
else:
    margem_bruta_col = 'margemBruta'

# Tratar casos onde 'margemBruta' pode ter valores inválidos
df[margem_bruta_col] = df[margem_bruta_col].replace(VALOR_AUSENTE, None)
df[margem_bruta_col] = df[margem_bruta_col].astype(float)

# Definir faixas para margemBruta
bins = [0.0, 0.2, 0.5, 0.8, 1.0]
labels = ['Muito Baixa', 'Baixa', 'Média', 'Alta']
df['faixa_margemBruta'] = pd.cut(df[margem_bruta_col], bins=bins, labels=labels, include_lowest=True)

# Tratar valores ausentes em 'faixa_margemBruta'
df['faixa_margemBruta'] = df['faixa_margemBruta'].cat.add_categories('Ausente').fillna('Ausente')

# Aplicar LabelEncoder em 'faixa_margemBruta'
label_encoder_margem_bruta = LabelEncoder()
df['classificacaoMargemBruta'] = label_encoder_margem_bruta.fit_transform(df['faixa_margemBruta'])

# Adicionar a categorização e codificação de maiorAtraso
if 'maiorAtraso' in df.columns:
    # Tratar valores faltantes
    df['maiorAtraso'] = df['maiorAtraso'].replace(VALOR_AUSENTE, None)
    df['maiorAtraso'] = df['maiorAtraso'].astype(float)
    # Criar faixas para 'maiorAtraso' usando quartis
    df['faixa_maiorAtraso'] = pd.qcut(df['maiorAtraso'], q=4, duplicates='drop',
                                      labels=['Sem Atraso', 'Atraso Baixo', 'Atraso Moderado', 'Atraso Alto'])
    # Tratar valores ausentes
    df['faixa_maiorAtraso'] = df['faixa_maiorAtraso'].cat.add_categories('Ausente').fillna('Ausente')

    # Aplicar LabelEncoder em 'faixa_maiorAtraso'
    label_encoder_maior_atraso = LabelEncoder()
    df['classificacao_maiorAtraso'] = label_encoder_maior_atraso.fit_transform(df['faixa_maiorAtraso'])
else:
    df['faixa_maiorAtraso'] = 'Ausente'
    df['classificacao_maiorAtraso'] = 0  # Placeholder

# Adicionar o tratamento de 'endividamento'
def classificar_endividamento(valor):
    if valor == VALOR_AUSENTE:
        return 'Ausente'
    elif valor <= 100000:
        return 'Baixo'
    elif valor <= 500000:
        return 'Médio'
    elif valor <= 1000000:
        return 'Alto'
    else:
        return 'Muito Alto'

df['classificacaoEndividamento'] = df['endividamento'].apply(classificar_endividamento)

# Aplicar LabelEncoder em 'classificacaoEndividamento'
label_encoder_endividamento = LabelEncoder()
df['classificacaoEndividamentoEncoded'] = label_encoder_endividamento.fit_transform(df['classificacaoEndividamento'])

# Adicionar o tratamento de 'faturamentoBruto'
def classificar_faturamento(valor):
    if valor == VALOR_AUSENTE:
        return 'Ausente'
    elif valor <= 500000:
        return 'Baixo'
    elif valor <= 2000000:
        return 'Médio'
    elif valor <= 5000000:
        return 'Alto'
    else:
        return 'Muito Alto'

df['classificacaoFaturamentoBruto'] = df['faturamentoBruto'].apply(classificar_faturamento)

# Aplicar LabelEncoder em 'classificacaoFaturamentoBruto'
label_encoder_faturamento = LabelEncoder()
df['classificacaoFaturamentoBrutoEncoded'] = label_encoder_faturamento.fit_transform(df['classificacaoFaturamentoBruto'])

# Adicionar o tratamento de 'capitalSocial'
def classificar_capital_social(valor):
    if valor == VALOR_AUSENTE:
        return 'Ausente'
    elif valor <= 50000:
        return 'Baixo'
    elif valor <= 200000:
        return 'Médio'
    elif valor <= 500000:
        return 'Alto'
    else:
        return 'Muito Alto'

df['classificacaoCapitalSocial'] = df['capitalSocial'].apply(classificar_capital_social)

# Aplicar LabelEncoder em 'classificacaoCapitalSocial'
label_encoder_capital_social = LabelEncoder()
df['classificacaoCapitalSocialEncoded'] = label_encoder_capital_social.fit_transform(df['classificacaoCapitalSocial'])

# Arredondar 'scorePontualidade' para uma casa decimal
df['scorePontualidade'] = df['scorePontualidade'].astype(float).round(1)

# Atualizar a lista de colunas para inserir
colunas_para_inserir = [
    'faixa_margemBruta', 'classificacaoMargemBruta',
    'faixa_maiorAtraso', 'classificacao_maiorAtraso',
    'classificacaoTotalPatrimonioLiquido', 'classificacaoTotalPatrimonioLiquidoEncoded',
    'classificacaoEndividamento', 'classificacaoEndividamentoEncoded',
    'classificacaoFaturamentoBruto', 'classificacaoFaturamentoBrutoEncoded',
    'classificacaoCapitalSocial', 'classificacaoCapitalSocialEncoded'
]

# Rearranjar as colunas para mover 'colunas_para_inserir' para o início
df = df.reindex(columns=colunas_para_inserir + [col for col in df.columns if col not in colunas_para_inserir])

# Visualizar as primeiras linhas para verificar
print(df.head())

# Salvar o arquivo transformado
df.to_csv('solicitacoescredito_transformado_final.csv', index=False)
print("Arquivo CSV final transformado com sucesso!")
