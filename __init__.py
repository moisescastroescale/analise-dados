"""
Pacote de Análise de Dados

Um pacote Python para análise exploratória e clusterização de dados.
"""

__version__ = "0.1.0"

import logging

from . import exploratoria
from . import clusterizacao
from . import tratamento
from .exploratoria import AnaliseExploratoria
from .tratamento import TratamentoDados

__all__ = ['exploratoria', 'clusterizacao', 'tratamento', 'AnaliseExploratoria', 'TratamentoDados', 'configurar_logging']


def configurar_logging(nivel=logging.INFO, formato=None, arquivo=None):
    """
    Configura o logging para o pacote analise_dados.
    
    Parameters
    ----------
    nivel : int, default logging.INFO
        Nível de logging (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
    formato : str, optional
        Formato das mensagens de log. Se None, usa formato padrão.
    arquivo : str, optional
        Caminho do arquivo para salvar os logs. Se None, logs vão para stdout.
    
    Examples
    --------
    >>> from analise_dados import configurar_logging
    >>> import logging
    >>> 
    >>> # Configurar para mostrar apenas INFO e acima
    >>> configurar_logging(nivel=logging.INFO)
    >>> 
    >>> # Configurar para salvar em arquivo
    >>> configurar_logging(nivel=logging.DEBUG, arquivo='analise.log')
    """
    if formato is None:
        formato = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = []
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(nivel)
    console_handler.setFormatter(logging.Formatter(formato))
    handlers.append(console_handler)
    
    # Handler para arquivo (se especificado)
    if arquivo:
        file_handler = logging.FileHandler(arquivo, encoding='utf-8')
        file_handler.setLevel(nivel)
        file_handler.setFormatter(logging.Formatter(formato))
        handlers.append(file_handler)
    
    # Configurar logger do pacote
    logger = logging.getLogger('analise_dados')
    logger.setLevel(nivel)
    logger.handlers = []  # Limpar handlers existentes
    for handler in handlers:
        logger.addHandler(handler)
    
    # Evitar propagação para o logger root
    logger.propagate = False

