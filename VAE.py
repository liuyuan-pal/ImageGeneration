import tensorflow as tf
import numpy as np
import math

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape):
    low = -np.sqrt(6.0/(shape[0] + shape[1]))
    high = np.sqrt(6.0/(shape[0] + shape[1]))
    weight=tf.Variable(tf.random_uniform((shape[0],shape[1]),
                     minval=low, maxval=high,
                     dtype=tf.float32),name='weight')
    # weight=tf.random_normal(shape, mean=0.0, stddev=0.1,name='weight')
    return weight

def bias_variable(shape):
    bias=tf.Variable(tf.zeros(shape),name='bias')
    return bias

def initialize_param(input_dims,hidden_nums,latent_num):
    encoder_params={}
    with tf.name_scope('en_h1'):
        encoder_params['w1']=weight_variable([input_dims,hidden_nums[0]])
        encoder_params['b1']=bias_variable([hidden_nums[0],])

    with tf.name_scope('en_h2'):
        encoder_params['w2']=weight_variable([hidden_nums[0],hidden_nums[1]])
        encoder_params['b2']=bias_variable([hidden_nums[1],])

    with tf.name_scope('en_h3'):
        encoder_params['w3']=weight_variable([hidden_nums[1],hidden_nums[2]])
        encoder_params['b3']=bias_variable([hidden_nums[2],])

    with tf.name_scope('z_var'):
        encoder_params['w_var']=weight_variable([hidden_nums[2],latent_num])
        encoder_params['b_var']=bias_variable([latent_num,])

    with tf.name_scope('z_mean'):
        encoder_params['w_mean']=weight_variable([hidden_nums[2],latent_num])
        encoder_params['b_mean']=bias_variable([latent_num,])

    decoder_params={}
    with tf.name_scope('de_h1'):
        decoder_params['w1']=weight_variable([latent_num,hidden_nums[2]])
        decoder_params['b1']=bias_variable([hidden_nums[2],])

    with tf.name_scope('de_h2'):
        decoder_params['w2']=weight_variable([hidden_nums[2],hidden_nums[1]])
        decoder_params['b2']=bias_variable([hidden_nums[1],])

    with tf.name_scope('de_h3'):
        decoder_params['w3'] = weight_variable([hidden_nums[1], hidden_nums[0]])
        decoder_params['b3'] = bias_variable([hidden_nums[0], ])

    with tf.name_scope('prob_x'):
        decoder_params['w_p']=weight_variable([hidden_nums[0],input_dims])
        decoder_params['b_p']=bias_variable([input_dims,])

    return encoder_params,decoder_params


def encoder(x,e,encoder_params):
    with tf.name_scope('en_h1'):
        h1=tf.matmul(x,encoder_params['w1'])+encoder_params['b1']
        h1=tf.nn.relu(h1,name='relu')

    with tf.name_scope('en_h2'):
        h2=tf.matmul(h1,encoder_params['w2'])+encoder_params['b2']
        h2=tf.nn.relu(h2,name='relu')

    with tf.name_scope('en_h3'):
        h3=tf.matmul(h2,encoder_params['w3'])+encoder_params['b3']
        h3=tf.nn.relu(h3,name='relu')

    with tf.name_scope('z_var'):
        z_log_var=tf.matmul(h3,encoder_params['w_var'])+encoder_params['b_var']
        variable_summaries(z_log_var)

    with tf.name_scope('b_var'):
        variable_summaries(encoder_params['b_var'])

    with tf.name_scope('w_var'):
        variable_summaries(encoder_params['w_var'])

    with tf.name_scope('z_mean'):
        z_mean=tf.matmul(h3,encoder_params['w_mean'])+encoder_params['b_mean']
        variable_summaries(z_mean)

    with tf.name_scope('z'):
        z=z_mean+tf.exp(0.5*z_log_var)*e
        variable_summaries(z)

    return z,z_log_var,z_mean

def decoder(z,decoder_params):
    with tf.name_scope('de_h1'):
        h1=tf.matmul(z,decoder_params['w1'])+decoder_params['b1']
        h1=tf.nn.relu(h1,name='relu')

    with tf.name_scope('de_h2'):
        h2=tf.matmul(h1,decoder_params['w2'])+decoder_params['b2']
        h2=tf.nn.relu(h2,name='relu')

    with tf.name_scope('de_h3'):
        h3=tf.matmul(h2,decoder_params['w3'])+decoder_params['b3']
        h3=tf.nn.relu(h3,name='relu')
        variable_summaries(h3)

    with tf.name_scope('prob_x'):
        p=tf.matmul(h3,decoder_params['w_p'])+decoder_params['b_p']
        p=tf.nn.sigmoid(p,name='sigmoid')
        variable_summaries(p)

    return p

def loss(x,z_mean,z_log_var,p,epsilon=1e-5):
    with tf.name_scope('kl_loss'):
        kl_loss=-(1.0+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))/2.0
        kl_loss=tf.reduce_sum(kl_loss,axis=1)
        variable_summaries(kl_loss)

    with tf.name_scope('prob_loss'):
        prob_loss=-(x*tf.log(epsilon+p)+(1.0-x)*tf.log(epsilon+1.0-p))
        prob_loss=tf.reduce_sum(prob_loss,axis=1)
        variable_summaries(prob_loss)

    total_loss=tf.add(kl_loss,prob_loss,name='total_loss')
    total_loss=tf.reduce_mean(total_loss)

    # tf.summary.scalar('loss',total_loss)

    return total_loss

def train_MNIST_VAE(hidden_nums,latent_num,batch_size,prefix,learning_rate,step_num):
    #load data
    from data_utils import load_MNIST
    X=load_MNIST()
    X=X.reshape(X.shape[0],-1)
    X=X>0
    X=X.astype(np.float32)
    input_dims=28**2

    sess=tf.InteractiveSession()
    # build graph
    encoder_params,decoder_params=initialize_param(input_dims,hidden_nums,latent_num)
    x=tf.placeholder(tf.float32,[None,input_dims])
    e=tf.placeholder(tf.float32,[None,latent_num])
    z,z_var,z_mean=encoder(x,e,encoder_params)
    p=decoder(z,decoder_params)
    total_loss=loss(x,z_mean,z_var,p)
    variable_summaries(total_loss)

    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(learning_rate,global_step,2000,0.9,staircase=True)
    # optimizer
    with tf.name_scope('train'):
        train_step=tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step=global_step)

    # init summery and variable
    merged=tf.summary.merge_all()
    summery_writer=tf.summary.FileWriter('log/'+prefix+'/')
    tf.global_variables_initializer().run()

    total_size=X.shape[0]
    cur=0
    epoch_num=0
    batch_num=0

    saver=tf.train.Saver()

    log_interval=100
    # train
    while True:
        sample_val=np.random.normal(0,1.0,[batch_size,latent_num])
        data_val=np.zeros([batch_size,input_dims])
        if cur+batch_size>total_size:
            data_val[:total_size-cur,:]=X[cur:,:]
            np.random.shuffle(X)
            data_val[total_size-cur:,:]=X[:cur+batch_size-total_size,:]
            cur+=batch_size-total_size

            # save every epoch
            epoch_num+=1
            saver.save(sess,'model/'+prefix+'.ckpt',global_step=epoch_num)

            batch_num=0

            if epoch_num>step_num:
                break

        else:
            data_val=X[cur:cur+batch_size,:]
            cur+=batch_size

        # data_val=tf.constant(data_val)
        # import matplotlib.pyplot as plt
        # plt.imshow(data_val[0].reshape([28,28]),cmap='gray')
        # plt.show()


        feed_dict={x:data_val,e:sample_val}
        # print(feed_dict)

        if batch_num%log_interval==0:
            _,loss_val,summary=sess.run([train_step,total_loss,merged],feed_dict)
            summery_writer.add_summary(summary)
            print('epoch {} batch {} loss {}'.format(epoch_num,batch_num,loss_val))
        else:
            sess.run(train_step,feed_dict)

        batch_num+=1

def reconstruct_x(X,hidden_num,latent_num,batch_size,prefix,epoch_num):
    input_dims=28*28
    encoder_params,decoder_params=initialize_param(input_dims,hidden_num,latent_num)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./model/"+prefix+".ckpt-{}".format(epoch_num))

        x = tf.placeholder(tf.float32, [None, input_dims])
        e = tf.placeholder(tf.float32, [None, latent_num])
        z, z_var, z_mean = encoder(x, e, encoder_params)
        p = decoder(z,decoder_params)
        # total_loss = loss(x, z_mean, z_var, p, sample_num)
        # total_loss = total_loss / batch_size
        # variable_summaries(total_loss)

        # Restore variables from disk.
        Es=np.random.normal(0,1.0,[X.shape[0], latent_num])
        prob_x=sess.run(p,{e:Es,x:X})

        for i in range(X.shape[0]):
            import matplotlib.pyplot as plt
            plt.subplot(211)
            plt.imshow(prob_x[i].reshape(28,28),cmap='gray')
            plt.subplot(212)
            plt.imshow(X[i,:].reshape(28,28),cmap='gray')
            # plt.('{}.png'.format(i))
            plt.show()

def sample_new_x(sample_num,hidden_num,latent_num,batch_size,prefix,epoch_num):
    input_dims=28*28
    encoder_params,decoder_params=initialize_param(input_dims,hidden_num,latent_num)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./model/"+prefix+".ckpt-{}".format(epoch_num))

        z = tf.placeholder(tf.float32, [None, latent_num])
        p = decoder(z,decoder_params)

        prob_x=sess.run(p,{z:np.random.normal(0,1,[sample_num,latent_num])})

        for i in range(sample_num):
            import cv2
            img=np.asarray(prob_x[i, :])
            img=img*255
            img=img.astype(np.uint8)
            img=img.reshape([28,28])
            cv2.imwrite('result/VAE_MNIST/{}.jpg'.format(i),img)

    return prob_x

if __name__=='__main__':
    train_MNIST_VAE([750,500,250],30,100,'VAE_MNIST',1e-3,100)

    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets('data',
    #                                     one_hot=True,
    #                                     fake_data=False)
    # print(mnist.train.next_batch(100))

    # from data_utils import load_MNIST
    # X=load_MNIST()
    # X=X.reshape(X.shape[0],-1)
    # X=X>0
    # X=X.astype(np.float32)
    # input_dims=28**2
    #
    # reconstruct_x(X[:30,:],[750,500,250],30,30,'VAE_MNIST')
    # sample_new_x(1000,[750,500,250],30,30,'VAE_MNIST',101)