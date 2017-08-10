from lenet.trainer import trainer
from lenet.network import lenet5      
from lenet.dataset import mnist

if __name__ == '__main__':
    dataset = mnist()   
    net = lenet5(images = dataset.images)  
    net.cook(labels = dataset.labels)
    bp = trainer (net, dataset.feed)
    bp.train()
    bp.train(   iter= 10000, 
                mini_batch_size = 500, 
                update_after_iter = 100,
                summarize = True   )