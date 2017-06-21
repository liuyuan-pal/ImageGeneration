import tensorflow as tf
import tensorflow.contrib.layers as L
import math
from tensorflow.examples.tutorials.mnist import input_data
import cv2

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def generator(z,initial_shape,target_shape,final_activate,init_fn,for_train=True):
    h=L.fully_connected(z,initial_shape[0]*initial_shape[1]*initial_shape[2],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=L.batch_norm,
                         normalizer_params={'is_training': for_train},
                         weights_initializer=init_fn)

    h=tf.reshape(h,[-1,initial_shape[0],initial_shape[1],initial_shape[2]])
    k=math.log(target_shape[0]/initial_shape[0],2)
    next_channel=int(initial_shape[2]/2)
    for i in range(int(k)):
        h=L.conv2d_transpose(inputs=h,num_outputs=int(next_channel),
                             kernel_size=3,stride=2,padding='same',
                             activation_fn=tf.nn.relu,
                             normalizer_fn=L.batch_norm,
                             normalizer_params={'is_training': for_train},
                             weights_initializer=init_fn)
        next_channel/=2

    out=L.conv2d_transpose(h,num_outputs=target_shape[2],
                           kernel_size=3,stride=1,
                           activation_fn=final_activate,
                           weights_initializer=init_fn)
    return out

def critic(img,init_channel,conv_layers,init_fn,reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()

        for _ in range(conv_layers):
            img = L.conv2d(img, num_outputs=init_channel,
                           kernel_size=3,stride=2,
                           activation_fn=tf.nn.relu,
                           normalizer_fn=L.batch_norm,
                           normalizer_params={'is_training': True},
                           weights_initializer=init_fn)

            init_channel*=2

        img=L.flatten(img)
        logit = L.fully_connected(img, 1, activation_fn=None)

    return logit

def build_train_graph(
    batch_size=100,
    latent_dims=64,
    lr_g=5e-5,
    lr_c=5e-5,
    clamp_lower=-0.01,
    clamp_upper=0.01
):

    z=tf.random_normal([batch_size,latent_dims])
    real_img=tf.placeholder(tf.float32,[batch_size,28,28,1])

    # MNIST generator
    with tf.variable_scope('generator'):
        generate_img=generator(z,[7,7,64],[28,28,1],tf.sigmoid,L.xavier_initializer(uniform=False))
    fake_logit=critic(generate_img,32,3,L.xavier_initializer(uniform=False))
    true_logit=critic(real_img,32,3,L.xavier_initializer(uniform=False),True)

    tf.summary.image('img',generate_img,max_outputs=10)

    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    c_loss=tf.reduce_mean(fake_logit-true_logit)
    g_loss=tf.reduce_mean(-fake_logit)

    tf.summary.scalar('c_loss',c_loss)

    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

    opt_g = L.optimize_loss(loss=g_loss,learning_rate=lr_g,
                            optimizer=tf.train.RMSPropOptimizer,
                             variables=theta_g, global_step=counter_g,
                             summaries=['gradient_norm'])
    opt_c =  L.optimize_loss(loss=c_loss, learning_rate=lr_c,
                             optimizer=tf.train.RMSPropOptimizer,
                             variables=theta_c, global_step=counter_c,
                             summaries = ['gradient_norm'])

    clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_c]
    # merge the clip operations on critic variables
    with tf.control_dependencies([opt_c]):
        opt_c = tf.tuple(clipped_var_c)

    return opt_g,opt_c,real_img,c_loss

import numpy as np

def train_WGAN_MNIST():
    sess=tf.InteractiveSession()
    dataset = input_data.read_data_sets('data', one_hot=True)
    batch_size=100
    opt_g,opt_c,real_img,c_loss=build_train_graph(batch_size)
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    prefix='WGAN__MNIST'
    converge_c_iter=100
    revise_c_iter=5
    max_iter_step=20000
    summary_writer = tf.summary.FileWriter('log/'+prefix+'/')
    tf.global_variables_initializer().run()



    def next_feed_dict():
        train_img = dataset.train.next_batch(batch_size)[0]
        train_img = np.reshape(train_img, (-1, 28, 28))
        train_img = np.expand_dims(train_img, -1)
        feed_dict = {real_img: train_img}

        return feed_dict

    fc=open('c_loss.log','w')
    fg=open('g_loss.log','w')
    log_interval=10
    for i in range(max_iter_step):
        if i < 25 or i % 500 == 0:
            citers = converge_c_iter
        else:
            citers = revise_c_iter

        for j in range(citers):
            feed_dict = next_feed_dict()
            if i % log_interval == log_interval-1 and j == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, merged,loss_val = sess.run([opt_c, merged_all,c_loss], feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged)
                summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(i), i)
                print('loss val {}'.format(loss_val))
                fc.write('loss val {}\n'.format(loss_val))
            else:
                sess.run(opt_c, feed_dict=feed_dict)

        # print('optimize g')
        feed_dict = next_feed_dict()
        if i % log_interval == log_interval - 1:
            _, merged = sess.run([opt_g, merged_all], feed_dict=feed_dict,options=run_options, run_metadata=run_metadata)
            summary_writer.add_summary(merged)
            summary_writer.add_run_metadata(run_metadata, 'generator_metadata {}'.format(i), i)
            print('generator optimize loss val {}'.format(loss_val))
            fg.write('loss val {}\n'.format(loss_val))
        else:
            sess.run(opt_g, feed_dict=feed_dict)

        if i % 1000 == 999:
            saver.save(sess, 'model/'+prefix+'/model.ckpt', global_step=i)

def generate_WGAN_MNIST(sample_num,restore_name,restore_epoch,latent_num=64):
    z=tf.placeholder(tf.float32,[sample_num,latent_num])
    with tf.variable_scope('generator'):
        g=generator(z,[7,7,64],[28,28,1],tf.sigmoid,L.xavier_initializer(uniform=False))

    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./model/"+restore_name+".ckpt-{}".format(restore_epoch))
        batch_z = np.random.normal(0, 1.0, [sample_num, latent_num]).astype(np.float32)
        rs = g.eval(feed_dict={z:batch_z})

    for i in range(sample_num):
        cv2.imwrite('result/WGAN_MNIST/{}.jpg'.format(i),np.asarray(rs[i]*255,dtype=np.uint8))



if __name__=="__main__":
    generate_WGAN_MNIST(100,'WGAN_MNIST/WGAN__MNISTmodel',17999)




