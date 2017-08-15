Trainer
=======

.. todo::

    This part of the tutorial is currently being done.

The trainer is perhaps the module that is most unique to tensorflow and is most different from theano.
Tensroflow uses :meth:`tf.Session` to parse computational graphs unlike in theano where we'd use 
the :meth:`theano.function` methods. 
For a detailed tutorial on how tensorflow processes and runs graphs, refer 
`this page <https://www.tensorflow.org/api_guides/python/client>`_.

The :class:`lenet.trainer.trainer` class takes as input an object of :class:`lenet.network.lenet5`
and :class:`lenet.dataset.mnist`. 
After adding them as attributes, it then initializes a new tensorflow session to run the 
computational graph and initializes all the variables in the graph.

.. code-block:: python

    self.network = network
    self.dataset = dataset 
    self.session = tf.InteractiveSession()        
    tf.global_variables_initializer().run()

The initializer class also calls the :meth:`lenet.trainer.trainer.summaries` method that initializes 
the summary writer (:meth:`tf.summary.FileWriter`) so that any processing on this computational graph
could be monitored at tensorboard. 

.. code-block:: python

    self.summary = tf.summary.merge_all()
    self.tensorboard = tf.summary.FileWriter("tensorboard")
    self.tensorboard.add_graph(self.session.graph)

.. warning::

    This section is still incomplete. There are more coming soon.. 

The trainer class documentation can be found in :ref:`trainer`. 