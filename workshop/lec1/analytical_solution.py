import numpy as np
import matplotlib.pyplot as plt

class regressor(object):
    """
    This is a sample class for lecture 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    def __init__(self, data):
        self.x, self.y = data        
        # Here is where your training and all the other magic should happen. 
        # Once trained you should have these parameters trained. 
        x = np.concatenate((np.ones((self.x.shape[0],1)), self.x), axis = 1)
        w = np.dot(np.linalg.pinv(np.dot(x.T,x)), np.dot(x.T,self.y))
        self.w = w[1:]
        self.b = w[0]
        
    def get_params (self):
        """ 
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return a random numpy array for demonstration purposes.

        """
        return (self.w, self.b)

    def get_predictions (self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x``                           
        """        
        predictions = np.add(np.dot(x, self.w), self.b)
        return predictions

    def plot(self, data = None, color = 'r'):
        """ Method will plot the line on an existing pyplot. 
        If data is provided, it will plot with the data. 
        
        Args:
            data: tuple of `(x,y)`. 
            color: What color the line should be default is red.
        """
        if data is not None:
            x, y = data
            plt.plot(x[:], y, 'bo')        
        plt.axis('equal')      
        plt.title(' Analytical Solution for Least Squares Linear Regression')  
        plt.title('Amazon Employee Compensation (Linear) Dataset')  
        plt.xlabel('Years of experience of the employee.')
        plt.ylabel('Compensation in $100,000')     
        grid = np.asarray([0, 10])[:,np.newaxis]
        predictions = self.get_predictions(grid)
        plt.plot(grid, predictions, color)
        plt.show()

class ridge_regressor(regressor):
    """
    This is a sample class for lecture 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
        alpha: Co-efficient for L2 regularizer.
                          
    """
    def __init__(self, data, alpha = 0.0001):
        self.x, self.y = data        
        # Here is where your training and all the other magic should happen. 
        # Once trained you should have these parameters trained. 
        x = np.concatenate((np.ones((self.x.shape[0],1)), self.x), axis = 1)
        w = np.dot(np.linalg.pinv(np.dot(x.T,x) + alpha*np.eye(x.shape[1])), np.dot(x.T,self.y))
        self.w = w[1:]
        self.b = w[0]
        
if __name__ == '__main__':
    pass 
