import numpy as np

def load_MNIST():
    with open('data/train-images.idx3-ubyte','rb') as f:
        f.seek(16)
        buffer=f.read()
        X=np.frombuffer(buffer,dtype=np.uint8)
    # print X.shape
    X=X.reshape([60000,28,28])
    return X