# Adapted from Avinash Kaitha for CSE591 taught by Ragav Venkatesan at ASU.
from analytical_solution import regressor
import numpy as np
from IPython import display
import matplotlib.pyplot as plt

class gd(regressor):
    """
    This is a sample class for GD.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.
        k: Number of folds of cross validation.

    """
    def feature_scaling(self,x):
        """
            Feature Scaling done on the x array (Min-max scaling)

            Args:
                x:  is a two or one dimensional ndarray ordered such that axis 0 is independent
                    data and data is spread along axis 1. If the array had only one dimension, it implies
                    that data is 1D.
        """
        return (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))


    def gradient_descent(self,x,y,w,alpha,beta,visualize = 100):
        """
            Gradient descent implementation including regularization.
            The loop stops when the total cost almost converges or after 10000 iterations
            Args:
                x:  is a two or one dimensional ndarray ordered such that axis 0 is independent
                    data and data is spread along axis 1. If the array had only one dimension, it implies
                    that data is 1D
                y:  is a 1D ndarray it will be of the same length as axis 0 or x.
                w:  Weights of the regression
                alpha: Learning Rate
                beta: Regularization Parameter
                visualize: number of iters after which to plot
        """
        prev_cost = 10
        curr_cost = self.get_cost(x,y,w)
        num_iters = 10000
        iter = 0
        while prev_cost - curr_cost > 0.00000001:

            gradient = ((x.T).dot(x.dot(w)-y) + beta*w.T.dot(w))/y.shape[0]
            w = w - alpha*gradient
            prev_cost = curr_cost
            curr_cost = self.get_cost(x, y, w)
            iter += 1
            if iter % visualize == 0:
                self.plot(params = w)
            if iter > num_iters:
                break              
        return w

    def plot(self, data = None, params = None, color = 'r'):
        """ Method will plot the line on an existing pyplot. 
        If data is provided, it will plot with the data. 
        
        Args:
            data: tuple of `(x,y)`. 
            color: What color the line should be default is red.
        """
        display.clear_output(wait=True)        
        if data is not None:
            x, y = data
        else:
            x, y = self.data
        plt.plot(x[:], y, 'bo')        
        plt.axis('equal')      
        plt.title(' Analytical Solution for Least Squares Linear Regression')  
        plt.title('Amazon Employee Compensation (Linear) Dataset')  
        plt.xlabel('Years of experience of the employee.')
        plt.ylabel('Compensation in $100,000')     
        grid = np.asarray([0, 10])[:,np.newaxis]
        if params is None:
            params = self.w
        predictions = self.get_predictions(grid, params = params)
        plt.plot(grid, predictions, color)
        plt.show()

    def get_cost(self,x,y,w):
        """
            Returns the total cost for a given set of weights
            Args:
                x: is a two or one dimensional ndarray ordered such that axis 0 is independent
                    data and data is spread along axis 1. If the array had only one dimension, it implies
                    that data is 1D.
                y:  is a 1D ndarray it will be of the same length as axis 0 or x.
                w:  Weights of the regression


        """
        cost = (((y-(x).dot(w)).T.dot(y-(x).dot(w)))/(2*y.shape[0]))[0][0]
        # print "Cost: ",cost
        return cost


    def cross_validate(self,x,y,k=1):
        """
            k fold cross validation performed with 70% data for training and 30% data for validation
            Sets the global weights parameter to the average of all the weights

            Args:
                x: is a two or one dimensional ndarray ordered such that axis 0 is independent
                    data and data is spread along axis 1. If the array had only one dimension, it implies
                    that data is 1D.
                y:  is a 1D ndarray it will be of the same length as axis 0 or x.
                k: Number of folds to crossvalidate

        """


        split_dim = int(0.7 * x.shape[0])
        # print "Split: ", split_dim
        fin_w = np.zeros((self.x.shape[1], 1))
        for i in range(0,k):
            w = np.random.rand(self.x.shape[1], 1)

            # print fin_w.shape
            rd = np.arange(len(x))
            np.random.shuffle(rd)
            x = x[rd]
            y = y[rd]
            x_train, x_test = x[:split_dim, :], x[split_dim:, :]
            y_train,y_test = y[:split_dim, :], y[split_dim:, :]

            fin_w = np.add(fin_w,self.gradient_descent(x_train, y_train, w, 0.01,0.05))
            
        self.w = np.divide(fin_w,float(k))
        return


    def __init__(self, data, k =1):
        self.data = data
        self.x, self.y = self.data
        self.b = np.random.rand(1)
        self.alpha = 0.01
        self.x = self.feature_scaling(self.x)
        one_col = np.ones((self.x.shape[0],1))
        self.x = np.append(one_col,self.x,1)
        self.w = np.random.rand(self.x.shape[1], 1)
        self.cross_validate(self.x,self.y)




    def get_params (self):
        """
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b).

        Notes:
            This code will return a random numpy array for demonstration purposes.

        """
        return (self.w[1:], self.w[0])

    def get_predictions (self, x, params = None):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.
            params: predict using these params.
        Returns:
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of
                            ``x``
        """
        x = self.feature_scaling(x)
        one_col = np.ones((x.shape[0], 1))
        x = np.append(one_col, x, 1)
        if params is None:
            return x.dot(self.w)
        else:
            return x.dot(params)

if __name__ == '__main__':
    pass
