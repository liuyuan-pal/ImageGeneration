import numpy as np
import matplotlib.pyplot as plt
import logging

class BinaryRestrictedBoltzmannMachine:
    '''
    X: [N,D]
    h: [N,H]
    W: [D,H]
    b: [H,]
    c: [D,]
    
    D visible variable number
    H hidden variable number
    N batch size
    
    '''
    def __init__(self,D,H,N):
        # initialization of parameters
        self.W=np.random.uniform(
            -4 * np.sqrt(6. / (H + D)),
            4 * np.sqrt(6. / (H + D)),
            [D,H]
        )
        # self.W=np.random.normal(0,1e-2,[D,H])
        self.b=np.zeros([H,])
        self.c=np.zeros([D,])
        self.N=N
        self.D=D
        self.H=H

        self.mW=np.zeros([D,H])
        self.mb=np.zeros([H,])
        self.mc=np.zeros([D,])

        pass

    def sample_X(self,h):
        MX=h.dot(self.W.transpose())+self.c
        MX=1.0/(1.0+np.exp(-MX))
        sample_val=np.random.rand(self.N,self.D)
        return sample_val<MX

    def sample_h(self,X):
        Mh=X.dot(self.W)+self.b
        Mh=1.0/(1.0+np.exp(-Mh))
        sample_val=np.random.rand(self.N,self.H)
        return sample_val<Mh

    def step(self,X,iter_num=15,learning_rate=0.1):
        sX=X
        h=self.sample_h(X)
        sh=h

        for _ in range(iter_num):
            sX = self.sample_X(sh)
            sh = self.sample_h(sX)


        Mh0=X.dot(self.W)+self.b
        Mh0=1.0/(1.0+np.exp(-Mh0))

        MhN=sX.dot(self.W)+self.b
        MhN=1.0/(1.0+np.exp(-MhN))

        MXN=sh.dot(self.W.transpose())+self.c
        MXN=1.0/(1.0+np.exp(-MXN))

        dW=(np.dot(X.transpose(),Mh0)-np.dot(MXN.transpose(),MhN))/float(self.N)
        db=np.mean(Mh0-MhN,axis=0)
        dc=np.mean(X-MXN,axis=0)

        # Mh0 = X.dot(self.W) + self.b
        # Mh0 = 1.0 / (1.0 + np.exp(-Mh0))
        # sample_val=np.random.rand(self.N,self.H)
        # h0=sample_val<Mh0
        #
        # MX1=h0.dot(self.W.transpose())+self.c
        # MX1=1.0/(1.0+np.exp(-MX1))
        # sample_val=np.random.rand(self.N,self.D)
        # X1=sample_val<MX1
        #
        #
        # Mh1 = X1.dot(self.W) + self.b
        # Mh1 = 1.0 / (1.0 + np.exp(-Mh1))
        # sample_val=np.random.rand(self.N,self.H)
        # h1=sample_val<Mh1

        # dW=(np.dot(X.transpose(),Mh0)-np.dot(MX1.transpose(),Mh1))/float(self.N)
        # db=np.mean(Mh0-Mh1,axis=0)
        # dc=np.mean(X-MX1,axis=0)

        self.W+=learning_rate*dW/self.D
        self.b+=learning_rate*db
        self.c+=learning_rate*dc

        return self.compute_pseudo_likelihood(X)
        # return np.sqrt(np.mean(((X1-X)**2)))

    def save_params(self,prefix,epoch):
        f=prefix+'_{}.params'.format(epoch)
        np.save(f,[self.W,self.b,self.c])

    def load_params(self,prefix,epoch):
        f=prefix+'_{}.params.npy'.format(epoch)
        W,b,c=np.load(f)
        self.W=W
        self.b=b
        self.c=c

    def reconstruct(self,X):
        Mh0 = X.dot(self.W) + self.b
        Mh0 = 1.0 / (1.0 + np.exp(-Mh0))
        sample_val=np.random.rand(self.N,self.H)
        h0=sample_val<Mh0

        MX1=h0.dot(self.W.transpose())+self.c
        MX1=1.0/(1.0+np.exp(-MX1))
        sample_val=np.random.rand(self.N,self.D)
        X1=sample_val<MX1

        return X1

    def sample(self,sample_num=20,iter_num=20):
        h=np.random.rand(sample_num,self.H)<0.5
        X=h.dot(self.W.transpose())+self.c
        X=1.0/(1.0+np.exp(-X))

        for i in range(iter_num):
            # X=np.random.rand(sample_num,self.D)<X
            h=self.sample_h(X)
            X=h.dot(self.W.transpose())+self.c
            X=1.0/(1.0+np.exp(-X))

        return X

    def free_energy(self,X):
        wx_b=X.dot(self.W) + self.b
        cx=X.dot(self.c)
        val=np.sum(np.log(1 + np.exp(wx_b)), axis=1)
        return -val-cx

    def compute_pseudo_likelihood(self,X):
        flip_index=np.random.choice(self.D,1)
        fx=self.free_energy(X)
        X_flip=np.copy(X)
        X_flip[:,flip_index]=np.logical_not(X_flip[:,flip_index])
        fx_flip=self.free_energy(X_flip)
        sigmd=1.0 / (1.0 + np.exp(fx - fx_flip))
        return self.D*np.log(sigmd)/self.N


class GaussianRestrictedBoltzmannMachine:
    '''
    X: [N,D]
    h: [N,H]
    W: [D,H]
    b: [H,]
    c: [D,]

    D visible variable number
    H hidden variable number
    N batch size

    '''

    def __init__(self, D, H, N):
        # initialization of parameters
        self.W = np.random.uniform(
            -4 * np.sqrt(6. / (H + D)),
            4 * np.sqrt(6. / (H + D)),
            [D, H]
        )
        # self.W=np.random.normal(0,1e-2,[D,H])
        self.b = np.zeros([H, ])
        self.c = np.zeros([D, ])
        self.N = N
        self.D = D
        self.H = H

        self.mW = np.zeros([D, H])
        self.mb = np.zeros([H, ])
        self.mc = np.zeros([D, ])

        pass

    def sample_X(self, h):
        MX = h.dot(self.W.transpose()) + self.c
        MX = 1.0 / (1.0 + np.exp(-MX))
        sample_val = np.random.rand(self.N, self.D)
        return sample_val < MX

    def sample_h(self, X):
        Mh = X.dot(self.W) + self.b
        Mh = 1.0 / (1.0 + np.exp(-Mh))
        sample_val = np.random.rand(self.N, self.H)
        return sample_val < Mh

    def step(self, X, iter_num=15, learning_rate=0.1):
        sX = X
        h = self.sample_h(X)
        sh = h

        for _ in range(iter_num):
            sX = self.sample_X(sh)
            sh = self.sample_h(sX)

        Mh0 = X.dot(self.W) + self.b
        Mh0 = 1.0 / (1.0 + np.exp(-Mh0))

        MhN = sX.dot(self.W) + self.b
        MhN = 1.0 / (1.0 + np.exp(-MhN))

        MXN = sh.dot(self.W.transpose()) + self.c
        MXN = 1.0 / (1.0 + np.exp(-MXN))

        dW = (np.dot(X.transpose(), Mh0) - np.dot(MXN.transpose(), MhN)) / float(self.N)
        db = np.mean(Mh0 - MhN, axis=0)
        dc = np.mean(X - MXN, axis=0)

        # Mh0 = X.dot(self.W) + self.b
        # Mh0 = 1.0 / (1.0 + np.exp(-Mh0))
        # sample_val=np.random.rand(self.N,self.H)
        # h0=sample_val<Mh0
        #
        # MX1=h0.dot(self.W.transpose())+self.c
        # MX1=1.0/(1.0+np.exp(-MX1))
        # sample_val=np.random.rand(self.N,self.D)
        # X1=sample_val<MX1
        #
        #
        # Mh1 = X1.dot(self.W) + self.b
        # Mh1 = 1.0 / (1.0 + np.exp(-Mh1))
        # sample_val=np.random.rand(self.N,self.H)
        # h1=sample_val<Mh1

        # dW=(np.dot(X.transpose(),Mh0)-np.dot(MX1.transpose(),Mh1))/float(self.N)
        # db=np.mean(Mh0-Mh1,axis=0)
        # dc=np.mean(X-MX1,axis=0)

        self.W += learning_rate * dW / self.D
        self.b += learning_rate * db
        self.c += learning_rate * dc

        return self.compute_pseudo_likelihood(X)
        # return np.sqrt(np.mean(((X1-X)**2)))

    def save_params(self, prefix, epoch):
        f = prefix + '_{}.params'.format(epoch)
        np.save(f, [self.W, self.b, self.c])

    def load_params(self, prefix, epoch):
        f = prefix + '_{}.params.npy'.format(epoch)
        W, b, c = np.load(f)
        self.W = W
        self.b = b
        self.c = c

    def reconstruct(self, X):
        Mh0 = X.dot(self.W) + self.b
        Mh0 = 1.0 / (1.0 + np.exp(-Mh0))
        sample_val = np.random.rand(self.N, self.H)
        h0 = sample_val < Mh0

        MX1 = h0.dot(self.W.transpose()) + self.c
        MX1 = 1.0 / (1.0 + np.exp(-MX1))
        sample_val = np.random.rand(self.N, self.D)
        X1 = sample_val < MX1

        return X1

    def sample(self, sample_num=20, iter_num=20):
        h = np.random.rand(sample_num, self.H) < 0.5
        X = h.dot(self.W.transpose()) + self.c
        X = 1.0 / (1.0 + np.exp(-X))

        for i in range(iter_num):
            # X=np.random.rand(sample_num,self.D)<X
            h = self.sample_h(X)
            X = h.dot(self.W.transpose()) + self.c
            X = 1.0 / (1.0 + np.exp(-X))

        return X

    def free_energy(self, X):
        wx_b = X.dot(self.W) + self.b
        cx = X.dot(self.c)
        val = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
        return -val - cx

    def compute_pseudo_likelihood(self, X):
        flip_index = np.random.choice(self.D, 1)
        fx = self.free_energy(X)
        X_flip = np.copy(X)
        X_flip[:, flip_index] = np.logical_not(X_flip[:, flip_index])
        fx_flip = self.free_energy(X_flip)
        sigmd = 1.0 / (1.0 + np.exp(fx - fx_flip))
        return self.D * np.log(sigmd) / self.N


def train_MNIST_RBM():

    from data_utils import load_MNIST

    prefix='model/rbm-MNIST_reinit'
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh=logging.FileHandler(prefix+'.log')
    fh.setLevel(logging.DEBUG)
    ch=logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    all_X=load_MNIST()
    all_X=all_X.reshape([60000,784])
    all_X.setflags(write=1)
    all_X=all_X>0

    batch_size=20
    total_size=all_X.shape[0]
    batch_num=total_size/batch_size

    epoch_num=50
    begin_epoch=20

    rbm=BinaryRestrictedBoltzmannMachine(784,500,batch_size)
    rbm.load_params(prefix,begin_epoch-1)

    for e in range(begin_epoch,epoch_num):
        np.random.shuffle(all_X)
        costs=[]
        for i in range(batch_num):
            cost=rbm.step(all_X[i * batch_size:(i + 1) * batch_size, :],15,1e-1)
            costs.append(cost)
            if i%500==0:
                logger.info('epoch {} batch {} costs {} |W| {} |b| {} |c| {}'.\
                    format(e,i,np.mean(costs),np.mean(rbm.W),np.mean(rbm.b),np.mean(rbm.c)))

        rbm.save_params(prefix,e)


        # plt.figure()
        # plt.imshow(X1[0,:].reshape(28,28),cmap='gray')
        # plt.figure()
        # plt.imshow(all_X[i * batch_size, :].reshape(28,28),cmap='gray')
        # plt.show()

        logger.info('epoch end {} costs {}!'.format(e,np.mean(costs)))

def reconstruct_MNIST_RBM():
    from data_utils import load_MNIST
    all_X=load_MNIST()
    all_X=all_X.reshape([60000,784])
    all_X.setflags(write=1)
    all_X=all_X>0

    batch_size=20
    # total_size=all_X.shape[0]
    # batch_num=total_size/batch_size

    prefix='model/rbm-MNIST_reinit'
    rbm=BinaryRestrictedBoltzmannMachine(784,500,20)
    rbm.load_params(prefix,6)

    np.random.shuffle(all_X)
    reconstuct_X=rbm.reconstruct(all_X[:batch_size])

    for i in range(batch_size):
        plt.figure()
        plt.subplot(211)
        plt.imshow(reconstuct_X[i,:].reshape(28,28),cmap='gray')
        plt.subplot(212)
        plt.imshow(all_X[i,:].reshape(28,28),cmap='gray')
        plt.show()

def sample_MNIST_RBM():
    sample_num=100
    prefix='model/rbm-MNIST_reinit'
    rbm=BinaryRestrictedBoltzmannMachine(784,500,sample_num)
    rbm.load_params(prefix,42)

    sample_X=rbm.sample(sample_num,iter_num=1000)

    for i in range(sample_num):
        # plt.figure()
        # plt.imshow(sample_X[i,:].reshape(28,28),cmap='gray')
        # plt.show()
        import cv2
        img=sample_X[i, :].reshape(28, 28)
        img*=255
        img=img.astype(np.uint8)
        cv2.imwrite('result/generated_3_{}.jpg'.format(i),img)


if __name__=="__main__":
    # train_MNIST_RBM()
    sample_MNIST_RBM()

