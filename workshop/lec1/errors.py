import numpy as np

def rmse ( a,  b ): 
    """
    This function produces a point-wise root mean squared error error between ``a`` and ``b``
    
    Args:
        a: first input ndarray
        b: second input ndarray

    Returns: 
        numpy float: rmse error 
    """        
    return np.sqrt(np.mean((a - b) ** 2))