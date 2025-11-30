# Análise de Dados

Pacote Python para análise exploratória e clusterização de dados.

## Instalação

```bash
pip install -e .
```

Ou instale as dependências diretamente:

```bash
pip install -r requirements.txt
```

## Estrutura do Pacote

```
analise-dados/
├── analise_dados/
│   ├── __init__.py
│   ├── tratamento.py      # Módulo de tratamento e limpeza de dados
│   ├── exploratoria.py    # Módulo de análise exploratória
│   └── clusterizacao.py   # Módulo de clusterização (a ser implementado)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Uso

### Configuração de Logging (Opcional)

O pacote usa logging para mensagens informativas. Você pode configurar o nível de logging:

```python
from analise_dados import configurar_logging
import logging

# Configurar para mostrar apenas INFO e acima (padrão)
configurar_logging(nivel=logging.INFO)

# Para debug detalhado
configurar_logging(nivel=logging.DEBUG)

# Para salvar logs em arquivo
configurar_logging(nivel=logging.INFO, arquivo='analise.log')

# Para desabilitar logs (apenas erros)
configurar_logging(nivel=logging.ERROR)
```

### Pipeline Completo de Análise

O fluxo recomendado é: **Tratamento → Análise Exploratória → Clusterização**

```python
import pandas as pd
from analise_dados import TratamentoDados, AnaliseExploratoria

# 1. Carregar dados brutos
df = pd.read_csv('dados_sujos.csv')

# 2. Tratar e limpar os dados
tratamento = TratamentoDados(df)
df_limpo = (tratamento
    .remover_duplicatas()
    .remover_colunas(['id', 'coluna_inutil'])
    .tratar_valores_ausentes(estrategia='media')
    .tratar_outliers(metodo='iqr', estrategia='limitar')
    .obter_dataframe())

# 3. Análise exploratória dos dados limpos
analise = AnaliseExploratoria(df_limpo)
analise.info_completo()
analise.plotar_valores_ausentes()
analise.matriz_correlacao()
```

### Tratamento de Dados

#### Usando a Classe (Recomendado)

```python
import pandas as pd
from analise_dados import TratamentoDados

# Carregar dados
df = pd.read_csv('dados.csv')

# Criar instância de tratamento
tratamento = TratamentoDados(df)

# Pipeline de tratamento encadeado
df_limpo = (tratamento
    .remover_duplicatas()
    .remover_colunas(['id', 'coluna_inutil'])
    .remover_colunas_com_muitos_ausentes(limite=0.5)
    .tratar_valores_ausentes(estrategia='media')
    .tratar_outliers(metodo='iqr', estrategia='limitar')
    .normalizar_colunas(metodo='standard')
    .codificar_categoricas(metodo='label')
    .obter_dataframe())

# Ver histórico de transformações
print(tratamento.historico)
```

#### Pipeline Completo com Configuração

```python
# Pipeline completo com configuração customizada
config = {
    'remover_duplicatas': True,
    'remover_colunas': ['id', 'timestamp'],
    'remover_colunas_ausentes': 0.7,
    'tratar_ausentes': {
        'estrategia': 'media',
        'colunas': ['coluna1', 'coluna2']
    },
    'tratar_outliers': {
        'metodo': 'iqr',
        'estrategia': 'limitar',
        'colunas': ['coluna1']
    },
    'normalizar': {
        'metodo': 'standard',
        'colunas': ['coluna1', 'coluna2']
    }
}

tratamento = TratamentoDados(df)
df_limpo = tratamento.pipeline_completo(config).obter_dataframe()
```

### Análise Exploratória

#### Usando a Classe (Recomendado)

A forma mais fácil de usar é através da classe `AnaliseExploratoria`:

```python
import pandas as pd
from analise_dados import AnaliseExploratoria

# Carregar seus dados
df = pd.read_csv('seu_arquivo.csv')

# Criar instância de análise
analise = AnaliseExploratoria(df)

# Agora você pode usar todos os métodos sem passar o DataFrame
analise.info_completo()  # Usa logging por padrão
analise.resumo_estatistico()
analise.valores_ausentes()
analise.plotar_valores_ausentes()
analise.matriz_correlacao()
analise.distribuicao_colunas()
analise.boxplot_colunas()
analise.detectar_outliers(metodo='iqr')

# Para usar print ao invés de logging (compatibilidade)
analise.info_completo(usar_logger=False)

# Acessar propriedades úteis
print(f"Dimensões: {analise.shape}")
print(f"Colunas numéricas: {analise.colunas_numericas}")
print(f"Colunas categóricas: {analise.colunas_categoricas}")
```

#### Usando Funções Diretamente

Você também pode usar as funções diretamente:

```python
import pandas as pd
from analise_dados import exploratoria

# Carregar seus dados
df = pd.read_csv('seu_arquivo.csv')

# Informações completas do DataFrame
exploratoria.info_completo(df)

# Resumo estatístico
exploratoria.resumo_estatistico(df)

# Valores ausentes
exploratoria.valores_ausentes(df)

# Visualizar valores ausentes
exploratoria.plotar_valores_ausentes(df)

# Matriz de correlação
exploratoria.matriz_correlacao(df)

# Distribuições
exploratoria.distribuicao_colunas(df)

# Boxplots
exploratoria.boxplot_colunas(df)

# Detectar outliers
exploratoria.detectar_outliers(df, metodo='iqr')
```

### Clusterização

O módulo de clusterização será implementado em breve.

## Funcionalidades

### Módulo Tratamento

#### Classe TratamentoDados

A classe `TratamentoDados` facilita a limpeza e pré-processamento de dados:

**Limpeza Básica:**
- **remover_duplicatas()**: Remove linhas duplicadas
- **remover_colunas()**: Remove colunas específicas
- **remover_colunas_com_muitos_ausentes()**: Remove colunas com muitos valores ausentes

**Tratamento de Valores Ausentes:**
- **tratar_valores_ausentes()**: Múltiplas estratégias (remover, media, mediana, moda, zero, valor, forward, backward)

**Tratamento de Outliers:**
- **tratar_outliers()**: Detecta e trata outliers (IQR ou Z-Score) com estratégias (remover, limitar, media, mediana)

**Normalização:**
- **normalizar_colunas()**: Padronização (StandardScaler) ou Normalização Min-Max

**Codificação:**
- **codificar_categoricas()**: Label Encoding ou One-Hot Encoding

**Outras Funcionalidades:**
- **converter_tipos()**: Conversão de tipos de dados
- **renomear_colunas()**: Renomeação de colunas
- **filtrar_linhas()**: Filtragem baseada em condições
- **pipeline_completo()**: Pipeline automatizado com configuração

**Propriedades:**
- **shape**: Dimensões do DataFrame
- **colunas_numericas**: Lista de colunas numéricas
- **colunas_categoricas**: Lista de colunas categóricas
- **historico**: Histórico de transformações aplicadas

### Módulo Exploratória

#### Classe AnaliseExploratoria

A classe `AnaliseExploratoria` encapsula o DataFrame e fornece métodos convenientes:

- **resumo_estatistico()**: Estatísticas descritivas completas
- **valores_ausentes()**: Análise de valores ausentes
- **plotar_valores_ausentes()**: Visualização de valores ausentes
- **matriz_correlacao()**: Cálculo e visualização de correlações
- **distribuicao_colunas()**: Histogramas de distribuição
- **boxplot_colunas()**: Boxplots para análise de outliers
- **detectar_outliers()**: Detecção de outliers (IQR ou Z-Score)
- **info_completo()**: Relatório completo do DataFrame

**Propriedades:**
- **shape**: Dimensões do DataFrame
- **colunas_numericas**: Lista de colunas numéricas
- **colunas_categoricas**: Lista de colunas categóricas

#### Funções Individuais

Todas as funções também estão disponíveis como funções independentes que aceitam um DataFrame como parâmetro.

## Requisitos

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.23.0
- plotly >= 5.0.0
- scikit-learn >= 1.0.0

## Licença

MIT

