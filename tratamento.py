"""
Módulo de Tratamento de Dados

Fornece funcionalidades para limpeza e pré-processamento de dados,
preparando-os para análise exploratória e modelagem.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict, Any, Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings
import logging

# Configurar logger para o módulo
logger = logging.getLogger(__name__)


class TratamentoDados:
    """
    Classe para tratamento e limpeza de dados.
    
    Facilita o pré-processamento de dados através de métodos encadeáveis
    que retornam um DataFrame limpo e pronto para análise.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from analise_dados.tratamento import TratamentoDados
    >>> 
    >>> df = pd.read_csv('dados_sujos.csv')
    >>> tratamento = TratamentoDados(df)
    >>> 
    >>> # Pipeline de tratamento
    >>> df_limpo = (tratamento
    ...     .remover_duplicatas()
    ...     .remover_colunas(['id', 'coluna_inutil'])
    ...     .tratar_valores_ausentes(estrategia='media')
    ...     .tratar_outliers(metodo='iqr')
    ...     .normalizar_colunas(['coluna1', 'coluna2'])
    ...     .obter_dataframe())
    """
    
    def __init__(self, df: pd.DataFrame, copiar: bool = True):
        """
        Inicializa o tratamento de dados com um DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a ser tratado
        copiar : bool, default True
            Se True, cria uma cópia do DataFrame. Se False, modifica o original.
        """
        self.df = df.copy() if copiar else df
        self._numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        self._historico = []  # Rastreia as transformações aplicadas
    
    @property
    def shape(self) -> tuple:
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
    
    @property
    def historico(self) -> List[str]:
        """Retorna o histórico de transformações aplicadas."""
        return self._historico.copy()
    
    def obter_dataframe(self) -> pd.DataFrame:
        """
        Retorna o DataFrame tratado.
        
        Returns
        -------
        pd.DataFrame
            DataFrame após todas as transformações aplicadas
        """
        return self.df.copy()
    
    def remover_duplicatas(self, subset: Optional[List[str]] = None, 
                          manter: str = 'first') -> 'TratamentoDados':
        """
        Remove linhas duplicadas do DataFrame.
        
        Parameters
        ----------
        subset : list, optional
            Lista de colunas para considerar na detecção de duplicatas.
            Se None, considera todas as colunas.
        manter : str, default 'first'
            Determina quais duplicatas manter ('first', 'last', False)
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        linhas_antes = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=manter)
        linhas_removidas = linhas_antes - len(self.df)
        
        self._historico.append(
            f"Removidas {linhas_removidas} linhas duplicadas"
        )
        return self
    
    def remover_colunas(self, colunas: Union[str, List[str]]) -> 'TratamentoDados':
        """
        Remove colunas do DataFrame.
        
        Parameters
        ----------
        colunas : str ou list
            Nome(s) da(s) coluna(s) a serem removidas
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        if isinstance(colunas, str):
            colunas = [colunas]
        
        colunas_existentes = [col for col in colunas if col in self.df.columns]
        if not colunas_existentes:
            warnings.warn("Nenhuma das colunas especificadas existe no DataFrame.")
            return self
        
        self.df = self.df.drop(columns=colunas_existentes)
        self._atualizar_listas_colunas()
        
        self._historico.append(
            f"Removidas colunas: {', '.join(colunas_existentes)}"
        )
        return self
    
    def remover_colunas_com_muitos_ausentes(self, limite: float = 0.5) -> 'TratamentoDados':
        """
        Remove colunas com porcentagem de valores ausentes acima do limite.
        
        Parameters
        ----------
        limite : float, default 0.5
            Porcentagem limite (0-1). Colunas com mais ausentes serão removidas.
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        missing_pct = (self.df.isnull().sum() / len(self.df)) > limite
        colunas_remover = missing_pct[missing_pct].index.tolist()
        
        if colunas_remover:
            self.df = self.df.drop(columns=colunas_remover)
            self._atualizar_listas_colunas()
            self._historico.append(
                f"Removidas {len(colunas_remover)} colunas com >{limite*100}% ausentes: "
                f"{', '.join(colunas_remover)}"
            )
        return self
    
    def tratar_valores_ausentes(self, estrategia: str = 'remover',
                               colunas: Optional[List[str]] = None,
                               valor_preencher: Optional[Any] = None) -> 'TratamentoDados':
        """
        Trata valores ausentes usando diferentes estratégias.
        
        Parameters
        ----------
        estrategia : str, default 'remover'
            Estratégia de tratamento:
            - 'remover': Remove linhas com valores ausentes
            - 'remover_colunas': Remove colunas com valores ausentes
            - 'media': Preenche com média (apenas numéricas)
            - 'mediana': Preenche com mediana (apenas numéricas)
            - 'moda': Preenche com moda (apenas categóricas)
            - 'zero': Preenche com zero
            - 'valor': Preenche com valor especificado em valor_preencher
            - 'forward': Forward fill
            - 'backward': Backward fill
        colunas : list, optional
            Lista de colunas específicas para tratar. Se None, trata todas.
        valor_preencher : any, optional
            Valor para preencher quando estrategia='valor'
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        if colunas is None:
            colunas = self.df.columns.tolist()
        else:
            colunas = [col for col in colunas if col in self.df.columns]
        
        if estrategia == 'remover':
            linhas_antes = len(self.df)
            self.df = self.df.dropna(subset=colunas)
            linhas_removidas = linhas_antes - len(self.df)
            self._historico.append(
                f"Removidas {linhas_removidas} linhas com valores ausentes"
            )
        
        elif estrategia == 'remover_colunas':
            colunas_remover = [col for col in colunas if self.df[col].isnull().all()]
            if colunas_remover:
                self.df = self.df.drop(columns=colunas_remover)
                self._atualizar_listas_colunas()
                self._historico.append(
                    f"Removidas colunas completamente vazias: {', '.join(colunas_remover)}"
                )
        
        elif estrategia == 'media':
            numeric_cols = [col for col in colunas if col in self._numeric_cols]
            for col in numeric_cols:
                media = self.df[col].mean()
                self.df[col] = self.df[col].fillna(media)
            self._historico.append(
                f"Preenchidos valores ausentes com média em: {', '.join(numeric_cols)}"
            )
        
        elif estrategia == 'mediana':
            numeric_cols = [col for col in colunas if col in self._numeric_cols]
            for col in numeric_cols:
                mediana = self.df[col].median()
                self.df[col] = self.df[col].fillna(mediana)
            self._historico.append(
                f"Preenchidos valores ausentes com mediana em: {', '.join(numeric_cols)}"
            )
        
        elif estrategia == 'moda':
            cat_cols = [col for col in colunas if col in self._categorical_cols]
            for col in cat_cols:
                moda = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else None
                if moda is not None:
                    self.df[col] = self.df[col].fillna(moda)
            self._historico.append(
                f"Preenchidos valores ausentes com moda em: {', '.join(cat_cols)}"
            )
        
        elif estrategia == 'zero':
            numeric_cols = [col for col in colunas if col in self._numeric_cols]
            self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
            self._historico.append(
                f"Preenchidos valores ausentes com zero em: {', '.join(numeric_cols)}"
            )
        
        elif estrategia == 'valor':
            if valor_preencher is None:
                raise ValueError("valor_preencher deve ser especificado quando estrategia='valor'")
            self.df[colunas] = self.df[colunas].fillna(valor_preencher)
            self._historico.append(
                f"Preenchidos valores ausentes com '{valor_preencher}' em: {', '.join(colunas)}"
            )
        
        elif estrategia == 'forward':
            self.df[colunas] = self.df[colunas].ffill()
            self._historico.append(f"Aplicado forward fill em: {', '.join(colunas)}")
        
        elif estrategia == 'backward':
            self.df[colunas] = self.df[colunas].bfill()
            self._historico.append(f"Aplicado backward fill em: {', '.join(colunas)}")
        
        else:
            raise ValueError(
                f"Estratégia '{estrategia}' não reconhecida. "
                f"Use: remover, remover_colunas, media, mediana, moda, zero, valor, forward, backward"
            )
        
        return self
    
    def tratar_outliers(self, metodo: str = 'iqr', 
                       colunas: Optional[List[str]] = None,
                       estrategia: str = 'remover') -> 'TratamentoDados':
        """
        Trata outliers nas colunas numéricas.
        
        Parameters
        ----------
        metodo : str, default 'iqr'
            Método de detecção ('iqr' ou 'zscore')
        colunas : list, optional
            Lista de colunas específicas. Se None, trata todas as numéricas.
        estrategia : str, default 'remover'
            Estratégia de tratamento:
            - 'remover': Remove linhas com outliers
            - 'limitar': Limita valores aos limites (capping)
            - 'media': Substitui outliers pela média
            - 'mediana': Substitui outliers pela mediana
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        if colunas is None:
            colunas = self._numeric_cols.copy()
        else:
            colunas = [col for col in colunas if col in self._numeric_cols]
        
        if not colunas:
            warnings.warn("Nenhuma coluna numérica encontrada para tratar outliers.")
            return self
        
        linhas_removidas = 0
        
        for col in colunas:
            data = self.df[col].dropna()
            
            if metodo == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            elif metodo == 'zscore':
                mean = data.mean()
                std = data.std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
            else:
                raise ValueError("Método deve ser 'iqr' ou 'zscore'")
            
            if estrategia == 'remover':
                mask = (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                linhas_antes = len(self.df)
                self.df = self.df[mask]
                linhas_removidas += (linhas_antes - len(self.df))
            
            elif estrategia == 'limitar':
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif estrategia == 'media':
                media = data.mean()
                self.df.loc[self.df[col] < lower_bound, col] = media
                self.df.loc[self.df[col] > upper_bound, col] = media
            
            elif estrategia == 'mediana':
                mediana = data.median()
                self.df.loc[self.df[col] < lower_bound, col] = mediana
                self.df.loc[self.df[col] > upper_bound, col] = mediana
        
        if estrategia == 'remover':
            self._historico.append(
                f"Removidas {linhas_removidas} linhas com outliers ({metodo})"
            )
        else:
            self._historico.append(
                f"Tratados outliers ({metodo}, {estrategia}) em: {', '.join(colunas)}"
            )
        
        return self
    
    def normalizar_colunas(self, colunas: Optional[List[str]] = None,
                          metodo: str = 'standard') -> 'TratamentoDados':
        """
        Normaliza/padroniza colunas numéricas.
        
        Parameters
        ----------
        colunas : list, optional
            Lista de colunas específicas. Se None, normaliza todas as numéricas.
        metodo : str, default 'standard'
            Método de normalização:
            - 'standard': Padronização (média=0, desvio=1)
            - 'minmax': Normalização Min-Max (0-1)
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        if colunas is None:
            colunas = self._numeric_cols.copy()
        else:
            colunas = [col for col in colunas if col in self._numeric_cols]
        
        if not colunas:
            warnings.warn("Nenhuma coluna numérica encontrada para normalizar.")
            return self
        
        if metodo == 'standard':
            scaler = StandardScaler()
            self.df[colunas] = scaler.fit_transform(self.df[colunas])
            self._historico.append(
                f"Padronização (StandardScaler) aplicada em: {', '.join(colunas)}"
            )
        elif metodo == 'minmax':
            scaler = MinMaxScaler()
            self.df[colunas] = scaler.fit_transform(self.df[colunas])
            self._historico.append(
                f"Normalização Min-Max aplicada em: {', '.join(colunas)}"
            )
        else:
            raise ValueError("Método deve ser 'standard' ou 'minmax'")
        
        return self
    
    def codificar_categoricas(self, colunas: Optional[List[str]] = None,
                             metodo: str = 'label') -> 'TratamentoDados':
        """
        Codifica variáveis categóricas em numéricas.
        
        Parameters
        ----------
        colunas : list, optional
            Lista de colunas específicas. Se None, codifica todas as categóricas.
        metodo : str, default 'label'
            Método de codificação:
            - 'label': Label Encoding (0, 1, 2, ...)
            - 'onehot': One-Hot Encoding (cria colunas binárias)
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        if colunas is None:
            colunas = self._categorical_cols.copy()
        else:
            colunas = [col for col in colunas if col in self._categorical_cols]
        
        if not colunas:
            warnings.warn("Nenhuma coluna categórica encontrada para codificar.")
            return self
        
        if metodo == 'label':
            le = LabelEncoder()
            for col in colunas:
                self.df[col] = le.fit_transform(self.df[col].astype(str))
            self._historico.append(
                f"Label Encoding aplicado em: {', '.join(colunas)}"
            )
            self._atualizar_listas_colunas()
        
        elif metodo == 'onehot':
            self.df = pd.get_dummies(self.df, columns=colunas, prefix=colunas)
            self._historico.append(
                f"One-Hot Encoding aplicado em: {', '.join(colunas)}"
            )
            self._atualizar_listas_colunas()
        
        else:
            raise ValueError("Método deve ser 'label' ou 'onehot'")
        
        return self
    
    def converter_tipos(self, mapeamento: Dict[str, str]) -> 'TratamentoDados':
        """
        Converte tipos de dados das colunas.
        
        Parameters
        ----------
        mapeamento : dict
            Dicionário com mapeamento {coluna: tipo}
            Tipos suportados: 'int', 'float', 'str', 'datetime', 'category'
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        for col, tipo in mapeamento.items():
            if col not in self.df.columns:
                warnings.warn(f"Coluna '{col}' não existe no DataFrame.")
                continue
            
            try:
                if tipo == 'datetime':
                    self.df[col] = pd.to_datetime(self.df[col])
                elif tipo == 'category':
                    self.df[col] = self.df[col].astype('category')
                else:
                    self.df[col] = self.df[col].astype(tipo)
            except Exception as e:
                warnings.warn(f"Erro ao converter '{col}' para {tipo}: {e}")
        
        self._atualizar_listas_colunas()
        self._historico.append(
            f"Conversão de tipos aplicada: {mapeamento}"
        )
        return self
    
    def renomear_colunas(self, mapeamento: Dict[str, str]) -> 'TratamentoDados':
        """
        Renomeia colunas do DataFrame.
        
        Parameters
        ----------
        mapeamento : dict
            Dicionário com mapeamento {nome_antigo: nome_novo}
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        self.df = self.df.rename(columns=mapeamento)
        self._atualizar_listas_colunas()
        self._historico.append(f"Colunas renomeadas: {mapeamento}")
        return self
    
    def filtrar_linhas(self, condicao: Callable) -> 'TratamentoDados':
        """
        Filtra linhas do DataFrame baseado em uma condição.
        
        Parameters
        ----------
        condicao : callable
            Função que recebe o DataFrame e retorna uma série booleana
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        linhas_antes = len(self.df)
        mask = condicao(self.df)
        self.df = self.df[mask]
        linhas_removidas = linhas_antes - len(self.df)
        
        self._historico.append(
            f"Filtradas {linhas_removidas} linhas baseado em condição"
        )
        return self
    
    def pipeline_completo(self, config: Optional[Dict[str, Any]] = None) -> 'TratamentoDados':
        """
        Aplica um pipeline completo de limpeza usando configuração padrão ou customizada.
        
        Parameters
        ----------
        config : dict, optional
            Dicionário com configurações do pipeline. Se None, usa configuração padrão.
            Chaves possíveis:
            - 'remover_duplicatas': bool
            - 'remover_colunas': list
            - 'remover_colunas_ausentes': float (limite 0-1)
            - 'tratar_ausentes': dict com estrategia, colunas, valor_preencher
            - 'tratar_outliers': dict com metodo, colunas, estrategia
            - 'normalizar': dict com metodo, colunas
            - 'codificar_categoricas': dict com metodo, colunas
        
        Returns
        -------
        TratamentoDados
            Instância atualizada para encadeamento de métodos
        """
        if config is None:
            config = {
                'remover_duplicatas': True,
                'tratar_ausentes': {'estrategia': 'media'},
                'tratar_outliers': {'metodo': 'iqr', 'estrategia': 'limitar'}
            }
        
        if config.get('remover_duplicatas', False):
            self.remover_duplicatas()
        
        if 'remover_colunas' in config:
            self.remover_colunas(config['remover_colunas'])
        
        if 'remover_colunas_ausentes' in config:
            self.remover_colunas_com_muitos_ausentes(config['remover_colunas_ausentes'])
        
        if 'tratar_ausentes' in config:
            params = config['tratar_ausentes']
            self.tratar_valores_ausentes(**params)
        
        if 'tratar_outliers' in config:
            params = config['tratar_outliers']
            self.tratar_outliers(**params)
        
        if 'normalizar' in config:
            params = config['normalizar']
            self.normalizar_colunas(**params)
        
        if 'codificar_categoricas' in config:
            params = config['codificar_categoricas']
            self.codificar_categoricas(**params)
        
        return self
    
    def _atualizar_listas_colunas(self):
        """Atualiza as listas de colunas numéricas e categóricas."""
        self._numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def __repr__(self) -> str:
        """Representação string da classe."""
        return (
            f"TratamentoDados(shape={self.shape}, "
            f"colunas_numericas={len(self._numeric_cols)}, "
            f"colunas_categoricas={len(self._categorical_cols)}, "
            f"transformacoes={len(self._historico)})"
        )

