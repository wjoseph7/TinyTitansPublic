import os

def get_TT_dir() -> str:
    """
    Summary:
        Returns absolute path to TT main directory
    Args:
        None
    Returns:
        str: with the absolute path to the TT directory
    """
    return os.path.dirname(__file__).replace('util', '') 

def get_CA_dir() -> str:
    """
    Summary:
        Returns absolute path to corporate actions data directory
    Args:
        None
    Returns:
        str: with the absolute path to the CA directory
    """

    TT_dir = get_TT_dir()
    CA_dir = TT_dir + 'data/corporate_actions/'

    return CA_dir