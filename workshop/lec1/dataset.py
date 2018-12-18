import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

class dataset_generator(object):
    """ 
    Class that creates a random dataset to be modelled by a linear regressor.
    
    Args:
        dimensions: number of dimensions of dataset (optional, default randomly 15-30)
        mu: mean of the gaussian with which we add noise (optional, default 0)
        sigma: variance of the gaussian with which we add noise (optional, default 0.5)
    """    
    def __init__(self, **kwargs):
        low = 15
        high = 30
        if 'dimensions' in kwargs.keys():
            self.dimensions = kwargs['dimensions']
        else:
            self.dimensions = np.random.randint(low = low,high = high)
        if 'mu' in kwargs.keys():
            self.mu = kwargs['mu']
        else:
            self.mu = 0
        if 'sigma' in kwargs.keys():
            self.sigma = kwargs['sigma']
        else:
            self.sigma = .5

        self.w = np.random.rand(self.dimensions,1)
        self.b = np.random.rand(1)

    def query_data(self, **kwargs):
        """
        Once initialized, this method will create more data.

        Args:
            samples: number of samples of data needed (optional, default randomly 10k - 50k)   
        Returns:
            tuple: data a tuple, ``(x,y)``
                ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
                data and data is spread along axis 1. 
                ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.                          
        """
        if 'samples' in kwargs.keys():
            samples = kwargs['samples']
        else:
            samples = np.random.randint(low = 1000, high = 5000)

        x = np.random.uniform(size = (samples, self.dimensions), low = 0, high = 10)        
        y = np.dot(x, self.w) + np.random.normal(self.mu, self.sigma, (samples,1)) + self.b
        
        return (x,y)

    def _create_noisy_samples(self, **kwargs):
        """
        Create samples in the opposite direction of the establishment.
        This is useful for demonstrating Regularization.
        
        Args:
            samples: number of samples of data needed (optional, default 1)   
        Returns:
            tuple: data a tuple, ``(x,y)``
                ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
                data and data is spread along axis 1. 
                ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.                          
        """
        if 'samples' in kwargs.keys():
            samples = kwargs['samples']
        else:
            samples = 1   
        if 'params' in kwargs.keys():
            w, b = kwargs['params']
        else:
            w = -1*self.w
            b = self.b - np.random.rand(1)
        x = np.random.uniform(size = (samples, self.dimensions), low = 0, high = 10)        
        y = np.dot(x, w) + np.random.normal(self.mu, self.sigma, (samples,1)) + b
        
        return (x,y)
    
    def query_noisy_data(self, **kwargs):
        """
        Once intitialized, this method will create noisy data along with good data.
        
        Args:
            samples: number of samples of real data needed (optional, default randomly 10k - 50k)
            noisy_samples: number of noisy samples to generate. default is 1.
            noisy_params (tuple): w and b of actual samples of noise.
            
        Returns:
            tuple: data a tuple, ``(x,y)``
                ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
                data and data is spread along axis 1. 
                ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.                          
        """
        x, y = self.query_data(**kwargs)
        if 'noisy_params' in kwargs.keys():
            noisy_params = kwargs['noisy_params']
        else:
            w = -1 * self.w
            b = -1 * self.b
            noisy_params = (w,b)
        if 'noisy_samples' in kwargs.keys():
            noisy_samples = kwargs['noisy_samples']
        else:
            noisy_samples = 1
        noisy_x, noisy_y = self._create_noisy_samples(samples = noisy_samples,
                                                      params = noisy_params)            
        x = np.concatenate((x,noisy_x), axis = 0)
        y = np.concatenate((y,noisy_y), axis = 0)
        return (x,y)
    
    def plot(self, x, y):
        """
        This method will plot the data as created by this dataset generator.
        Args:
            x: as produced by the ``query_data`` method's first element.
            y: as produced by the ``query_data`` method's second element.
        """
        plt.plot(x[:], y, 'bo')        
        plt.axis('equal')      
        plt.title('Amazon Employee Compensation (Linear) Dataset.')  
        plt.xlabel('Years of experience of the employee.')
        plt.ylabel('Compensation in $100,000.')
        plt.show()

    def _demo (self, samples = 50):
        """
        This is a demonstration method that will plot a version of a random dataset on the screen.
        
        Args:
            samples: number of samples of data needed (optional, default 20)         
        """        
        x, y = self.query_data(samples = samples) 
        self.plot(x, y)    