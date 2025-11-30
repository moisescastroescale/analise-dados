"""
Módulo de Análise Exploratória de Dados

Fornece funções para análise exploratória de dados, incluindo:
- Estatísticas descritivas
- Visualizações
- Detecção de valores ausentes
- Análise de correlação
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple
import logging

# Configurar logger para o módulo
logger = logging.getLogger(__name__)


class AnaliseExploratoria:
    """
    Classe para análise exploratória de dados.
    
    Facilita a análise de um DataFrame encapsulando-o e fornecendo
    métodos convenientes para análise exploratória.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from analise_dados.exploratoria import AnaliseExploratoria
    >>> 
    >>> df = pd.read_csv('dados.csv')
    >>> analise = AnaliseExploratoria(df)
    >>> analise.info_completo()
    >>> analise.resumo_estatistico()
    >>> analise.plotar_valores_ausentes()
    >>> analise.matriz_correlacao()
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa a análise exploratória com um DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a ser analisado
        """
        self.df = df.copy()
        self._numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Retorna as dimensões do DataFrame (linhas, colunas)."""
        return self.df.shape
    
    @property
    def colunas_numericas(self) -> List[str]:
        """Retorna lista de colunas numéricas."""
        return self._numeric_cols.copy()
    
    @property
    def colunas_categoricas(self) -> List[str]:
        """Retorna lista de colunas categóricas."""
        return self._categorical_cols.copy()
    
    def resumo_estatistico(self, incluir_categoricas: bool = False) -> pd.DataFrame:
        """
        Gera um resumo estatístico completo do DataFrame.
        
        Parameters
        ----------
        incluir_categoricas : bool, default False
            Se True, inclui colunas categóricas no resumo
        
        Returns
        -------
        pd.DataFrame
            DataFrame com estatísticas descritivas
        """
        return resumo_estatistico(self.df, incluir_categoricas)
    
    def valores_ausentes(self, porcentagem: bool = True) -> pd.Series:
        """
        Retorna a contagem ou porcentagem de valores ausentes por coluna.
        
        Parameters
        ----------
        porcentagem : bool, default True
            Se True, retorna porcentagem; se False, retorna contagem absoluta
        
        Returns
        -------
        pd.Series
            Série com valores ausentes por coluna
        """
        return valores_ausentes(self.df, porcentagem)
    
    def plotar_valores_ausentes(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plota um gráfico de barras mostrando valores ausentes por coluna.
        
        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Tamanho da figura
        """
        plotar_valores_ausentes(self.df, figsize)
    
    def matriz_correlacao(self, metodo: str = 'pearson', 
                         figsize: Tuple[int, int] = (12, 10), 
                         annot: bool = True) -> pd.DataFrame:
        """
        Calcula e visualiza a matriz de correlação.
        
        Parameters
        ----------
        metodo : str, default 'pearson'
            Método de correlação ('pearson', 'kendall', 'spearman')
        figsize : tuple, default (12, 10)
            Tamanho da figura
        annot : bool, default True
            Se True, mostra os valores de correlação no heatmap
        
        Returns
        -------
        pd.DataFrame
            Matriz de correlação
        """
        return matriz_correlacao(self.df, metodo, figsize, annot)
    
    def distribuicao_colunas(self, colunas: Optional[List[str]] = None, 
                            figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plota histogramas para colunas numéricas.
        
        Parameters
        ----------
        colunas : list, optional
            Lista de colunas específicas para plotar. Se None, plota todas as numéricas
        figsize : tuple, default (15, 10)
            Tamanho da figura
        """
        distribuicao_colunas(self.df, colunas, figsize)
    
    def boxplot_colunas(self, colunas: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plota boxplots para colunas numéricas.
        
        Parameters
        ----------
        colunas : list, optional
            Lista de colunas específicas para plotar. Se None, plota todas as numéricas
        figsize : tuple, default (15, 10)
            Tamanho da figura
        """
        boxplot_colunas(self.df, colunas, figsize)
    
    def detectar_outliers(self, metodo: str = 'iqr', 
                         colunas: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detecta outliers nas colunas numéricas.
        
        Parameters
        ----------
        metodo : str, default 'iqr'
            Método de detecção ('iqr' para Interquartile Range ou 'zscore' para Z-Score)
        colunas : list, optional
            Lista de colunas específicas. Se None, analisa todas as numéricas
        
        Returns
        -------
        pd.DataFrame
            DataFrame com informações sobre outliers detectados
        """
        return detectar_outliers(self.df, metodo, colunas)
    
    def info_completo(self, usar_logger: bool = True) -> None:
        """
        Exibe informações completas sobre o DataFrame.
        
        Parameters
        ----------
        usar_logger : bool, default True
            Se True, usa logging. Se False, usa print (para compatibilidade)
        """
        info_completo(self.df, usar_logger)
    
    def __repr__(self) -> str:
        """Representação string da classe."""
        return f"AnaliseExploratoria(shape={self.shape}, colunas_numericas={len(self._numeric_cols)}, colunas_categoricas={len(self._categorical_cols)})"


def resumo_estatistico(df: pd.DataFrame, incluir_categoricas: bool = False) -> pd.DataFrame:
    """
    Gera um resumo estatístico completo do DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado
    incluir_categoricas : bool, default False
        Se True, inclui colunas categóricas no resumo
    
    Returns
    -------
    pd.DataFrame
        DataFrame com estatísticas descritivas
    """
    if incluir_categoricas:
        return df.describe(include='all')
    return df.describe()


def valores_ausentes(df: pd.DataFrame, porcentagem: bool = True) -> pd.Series:
    """
    Retorna a contagem ou porcentagem de valores ausentes por coluna.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado
    porcentagem : bool, default True
        Se True, retorna porcentagem; se False, retorna contagem absoluta
    
    Returns
    -------
    pd.Series
        Série com valores ausentes por coluna
    """
    missing = df.isnull().sum()
    if porcentagem:
        return (missing / len(df)) * 100
    return missing


def plotar_valores_ausentes(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plota um gráfico de barras mostrando valores ausentes por coluna.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado
    figsize : tuple, default (10, 6)
        Tamanho da figura (width, height) em pixels
    """
    missing = valores_ausentes(df, porcentagem=True)
    missing = missing[missing > 0].sort_values(ascending=True)
    
    if len(missing) == 0:
        logger.info("Não há valores ausentes no DataFrame.")
        return
    
    fig = go.Figure(data=go.Bar(
        x=missing.values,
        y=missing.index,
        orientation='h',
        marker=dict(color=missing.values, colorscale='Reds', showscale=True)
    ))
    
    fig.update_layout(
        title='Valores Ausentes por Coluna',
        xaxis_title='Porcentagem de Valores Ausentes (%)',
        yaxis_title='Colunas',
        width=figsize[0] * 100,
        height=figsize[1] * 100,
        hovermode='closest'
    )
    
    fig.show()


def matriz_correlacao(df: pd.DataFrame, metodo: str = 'pearson', 
                     figsize: Tuple[int, int] = (12, 10), 
                     annot: bool = True) -> pd.DataFrame:
    """
    Calcula e visualiza a matriz de correlação.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com apenas colunas numéricas
    metodo : str, default 'pearson'
        Método de correlação ('pearson', 'kendall', 'spearman')
    figsize : tuple, default (12, 10)
        Tamanho da figura (width, height) em pixels
    annot : bool, default True
        Se True, mostra os valores de correlação no heatmap
    
    Returns
    -------
    pd.DataFrame
        Matriz de correlação
    """
    # Seleciona apenas colunas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("O DataFrame não contém colunas numéricas.")
    
    corr_matrix = numeric_df.corr(method=metodo)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2) if annot else None,
        texttemplate='%{text}' if annot else None,
        textfont={"size": 10},
        colorbar=dict(title="Correlação")
    ))
    
    fig.update_layout(
        title='Matriz de Correlação',
        width=figsize[0] * 100,
        height=figsize[1] * 100,
        xaxis_title="",
        yaxis_title=""
    )
    
    fig.show()
    
    return corr_matrix


def distribuicao_colunas(df: pd.DataFrame, colunas: Optional[List[str]] = None, 
                         figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plota histogramas para colunas numéricas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado
    colunas : list, optional
        Lista de colunas específicas para plotar. Se None, plota todas as numéricas
    figsize : tuple, default (15, 10)
        Tamanho da figura (width, height) em pixels
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("O DataFrame não contém colunas numéricas.")
    
    if colunas:
        numeric_df = numeric_df[colunas]
    
    n_cols = len(numeric_df.columns)
    n_rows = (n_cols + 2) // 3
    
    fig = make_subplots(
        rows=n_rows, 
        cols=3,
        subplot_titles=[f'Distribuição de {col}' for col in numeric_df.columns],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, col in enumerate(numeric_df.columns):
        row = (idx // 3) + 1
        col_pos = (idx % 3) + 1
        
        data = numeric_df[col].dropna()
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name=col,
                showlegend=False
            ),
            row=row,
            col=col_pos
        )
        
        fig.update_xaxes(title_text=col, row=row, col=col_pos)
        fig.update_yaxes(title_text="Frequência", row=row, col=col_pos)
    
    fig.update_layout(
        title_text="Distribuições das Colunas Numéricas",
        height=figsize[1] * 100,
        width=figsize[0] * 100
    )
    
    fig.show()


def boxplot_colunas(df: pd.DataFrame, colunas: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plota boxplots para colunas numéricas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado
    colunas : list, optional
        Lista de colunas específicas para plotar. Se None, plota todas as numéricas
    figsize : tuple, default (15, 10)
        Tamanho da figura (width, height) em pixels
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("O DataFrame não contém colunas numéricas.")
    
    if colunas:
        numeric_df = numeric_df[colunas]
    
    n_cols = len(numeric_df.columns)
    n_rows = (n_cols + 2) // 3
    
    fig = make_subplots(
        rows=n_rows, 
        cols=3,
        subplot_titles=[f'Boxplot de {col}' for col in numeric_df.columns],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, col in enumerate(numeric_df.columns):
        row = (idx // 3) + 1
        col_pos = (idx % 3) + 1
        
        data = numeric_df[col].dropna()
        fig.add_trace(
            go.Box(
                y=data,
                name=col,
                showlegend=False
            ),
            row=row,
            col=col_pos
        )
        
        fig.update_yaxes(title_text=col, row=row, col=col_pos)
    
    fig.update_layout(
        title_text="Boxplots das Colunas Numéricas",
        height=figsize[1] * 100,
        width=figsize[0] * 100
    )
    
    fig.show()


def detectar_outliers(df: pd.DataFrame, metodo: str = 'iqr', 
                      colunas: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Detecta outliers nas colunas numéricas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado
    metodo : str, default 'iqr'
        Método de detecção ('iqr' para Interquartile Range ou 'zscore' para Z-Score)
    colunas : list, optional
        Lista de colunas específicas. Se None, analisa todas as numéricas
    
    Returns
    -------
    pd.DataFrame
        DataFrame com informações sobre outliers detectados
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("O DataFrame não contém colunas numéricas.")
    
    if colunas:
        numeric_df = numeric_df[colunas]
    
    outliers_info = []
    
    for col in numeric_df.columns:
        data = numeric_df[col].dropna()
        
        if metodo == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        elif metodo == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > 3]
        else:
            raise ValueError("Método deve ser 'iqr' ou 'zscore'")
        
        outliers_info.append({
            'Coluna': col,
            'Total Outliers': len(outliers),
            'Porcentagem': (len(outliers) / len(data)) * 100
        })
    
    return pd.DataFrame(outliers_info)


def info_completo(df: pd.DataFrame, usar_logger: bool = True) -> None:
    """
    Exibe informações completas sobre o DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado
    usar_logger : bool, default True
        Se True, usa logging. Se False, usa print (para compatibilidade)
    """
    output_func = logger.info if usar_logger else print
    
    output_func("=" * 80)
    output_func("INFORMAÇÕES DO DATAFRAME")
    output_func("=" * 80)
    output_func(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    output_func(f"\nUso de memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    output_func("\n" + "=" * 80)
    output_func("TIPOS DE DADOS")
    output_func("=" * 80)
    output_func(f"\n{df.dtypes}")
    
    output_func("\n" + "=" * 80)
    output_func("VALORES AUSENTES")
    output_func("=" * 80)
    missing = valores_ausentes(df, porcentagem=True)
    if missing.sum() > 0:
        missing_str = str(missing[missing > 0].sort_values(ascending=False))
        output_func(f"\n{missing_str}")
    else:
        output_func("\nNenhum valor ausente encontrado.")
    
    output_func("\n" + "=" * 80)
    output_func("ESTATÍSTICAS DESCRITIVAS")
    output_func("=" * 80)
    stats = resumo_estatistico(df)
    output_func(f"\n{stats}")
    
    output_func("\n" + "=" * 80)
    output_func("PRIMEIRAS LINHAS")
    output_func("=" * 80)
    output_func(f"\n{df.head()}")

