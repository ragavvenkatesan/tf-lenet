# TF-Lenet

From being a long time user of Theano, I am now migrating to TensorFlow. 
This is my first implementation in TensorFlow and it is a typical Lenet-5 network trained to classify MNIST dataset. 
This is a simple code that is found in the tutorial that comes with TensorFlow.
I made it a little modularized so that it could be re-purposed for other networks later.
The documentation website that comes along with this repository helps users migrating from theano to tensorflow.

Obviously, you'd need [tensorflow](https://www.tensorflow.org/install/) and [numpy](https://docs.scipy.org/doc/numpy/user/install.html) installed. To run the code simply run:

```bash
python main.py
```

Once run, you can view all summaries at the tensorboard by running:

```bash
tensorboard --logdir tensorboard
```

Refer to the documentation site for more details.
Thanks for checking out the code, hope it was useful.

Ragav Venkatesan

http://www.ragav.net