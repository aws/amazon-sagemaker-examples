import random 
import numpy as np
import math
import matplotlib.pyplot as plt

def sample_from_test_set(test_imgs, sample_size=16):
    """Sample data from MNIST test set
    
    Args:
        test_imgs (numpy.ndarray): Test data in numpy array.
            Axis 0 needs to tbe the batch dimension.
        sample_size (int): number of test images to sample
    
    Returns:
        numpy.ndarray: selected test images as a numpy array. 
        It has shape (`sample_size`, 28, 28)
    """
    
    mask = random.sample(range(0, len(test_imgs)), sample_size)
    mask = np.array(mask, dtype=np.int8)
    return test_imgs[mask]
    

def inspect_images(images):
    """Display selected test images from MNIST
    
    Args:
        images (numpy.ndarray): Test data in numpy arra.
            Remove the channel depth dimension, 
            i.e it needs to have shape (`sample_size`, 28, 28)
            
    Returns:
        None
    """
    sample_size = images.shape[0]
    
    # display 16 images each row
    nrows = math.ceil(float(sample_size) / 16)
    fig, axs = plt.subplots(nrows=nrows, ncols=16, figsize=(16, nrows))
    
    if nrows==1:
        for i, row in enumerate(axs):
            if i < sample_size:
                row.imshow(images[i])
    else:
        for i, row in enumerate(axs):
            for j, col in enumerate(row):
                ix = i*16 + j
                if ix < sample_size:
                    col.imshow(images[ix])
    return    