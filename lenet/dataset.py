from tensorflow.examples.tutorials.mnist import input_data as mnist_feeder
import tensorflow as tf

class mnist(object):
    """
    Class for the mnist objects
    
    Args: 
        dir: Directory to cache at

    Attributes:
    
        These are variables of the class that are available outside. 
        
        *   ``images``: This is the placeholder for images. This needs to be fed in.
        *   ``labels``: This is the placeholder for images. This needs to be fed in.     
        *   ``feed``: This is a feeder from mnist tutorials of tensorflow.      
    """
    def __init__ (self, dir = 'data'):
        """
        Class constructor               
        """
        self.feed = mnist_feeder.read_data_sets (dir, one_hot = True)

        #Placeholders
        with tf.variable_scope('dataset_inputs') as scope:
            self.images = tf.placeholder(tf.float32, shape=[None, 784], name = 'images')
            self.labels = tf.placeholder(tf.float32, shape = [None, 10], name = 'labels') 