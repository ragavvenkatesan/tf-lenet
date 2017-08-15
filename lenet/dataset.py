from tensorflow.examples.tutorials.mnist import input_data as mnist_feeder
import tensorflow as tf

class mnist(object):
    """
    Class for the mnist objects
    
    Args: 
        dir: Directory to cache at

    Attributes:        
        images: This is the placeholder for images. This needs to be fed in using ``fed_dict``.
        labels: This is the placeholder for images. This needs to be fed in using ``fed_dict``.     
        feed: This is a feeder from mnist tutorials of tensorflow. Use this for feeding in data.      
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