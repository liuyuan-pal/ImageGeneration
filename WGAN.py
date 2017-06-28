import tensorflow as tf
import tensorflow.contrib.layers as L
import math
from tensorflow.examples.tutorials.mnist import input_data
import cv2

def variable_summaries(var,name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


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

def build_train_MNIST_graph(
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

def train_WGAN_MNIST():
    sess=tf.InteractiveSession()
    dataset = input_data.read_data_sets('data', one_hot=True)
    batch_size=100
    opt_g,opt_c,real_img,c_loss=build_train_MNIST_graph(batch_size)
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    prefix='WGAN_MNIST'
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

def build_train_MSR_face_graph(
    real_img,
    batch_size=64,
    latent_dims=1024,
    lr_g=5e-5,
    lr_c=5e-5,
    clamp_lower=-0.01,
    clamp_upper=0.01
):

    z=tf.random_normal([batch_size,latent_dims])
    # real_img=tf.placeholder(tf.float32,[batch_size,28,28,1])

    # face generator
    with tf.variable_scope('generator'):
        generate_img=generator(z,[4,4,128],[64,64,3],tf.tanh,L.xavier_initializer(uniform=False))
    fake_logit=critic(generate_img,32,4,L.xavier_initializer(uniform=False))
    true_logit=critic(real_img,32,4,L.xavier_initializer(uniform=False),True)

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

    return opt_g,opt_c,c_loss

import numpy as np

def train_WGAN_MSR_face():
    prefix='WGAN_Emotion'
    batch_size=100
    converge_c_iter=300
    converge_g_iter=50
    revise_c_iter=10
    max_iter_step=40000


    from data_utils import get_msr_train_faces
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement=True

    sess=tf.InteractiveSession()
    real_img,labels=get_msr_train_faces(batch_size)
    real_img=tf.image.resize_images(real_img,tf.constant([64,64],tf.int32))
    opt_g,opt_c,c_loss=build_train_MSR_face_graph(real_img=real_img,
                                                  batch_size=batch_size,latent_dims=1024,
                                                  lr_g=5e-5,lr_c=5e-5,
                                                  clamp_lower=-0.01,clamp_upper=0.01)


    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('log/'+prefix+'/')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tf.global_variables_initializer().run()


    fc=open('c_loss.log','w')
    fg=open('g_loss.log','w')
    log_interval=1
    for i in range(max_iter_step):
        if i < converge_g_iter or i % 500 == 0:
            citers = converge_c_iter
        else:
            citers = revise_c_iter

        for j in range(citers):
            if i % log_interval == log_interval-1 and j == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, merged,loss_val = sess.run([opt_c, merged_all,c_loss],
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged)
                summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(i), i)
                print('loss val {}'.format(loss_val))
                fc.write('loss val {}\n'.format(loss_val))
            else:
                sess.run(opt_c)

        # print('optimize g')
        if i % log_interval == log_interval - 1:
            _, merged = sess.run([opt_g, merged_all],options=run_options, run_metadata=run_metadata)
            summary_writer.add_summary(merged)
            summary_writer.add_run_metadata(run_metadata, 'generator_metadata {}'.format(i), i)
            print('generator optimize loss val {}'.format(loss_val))
            fg.write('loss val {}\n'.format(loss_val))
        else:
            sess.run(opt_g)

        if i % 1000 == 999:
            saver.save(sess, 'model/'+prefix+'/model.ckpt', global_step=i)

    coord.request_stop()
    coord.join(threads)
    sess.close()


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def average_loss(tower_losses):
    tower_losses=tf.stack(axis=0,values=tower_losses)
    result_loss=tf.reduce_mean(tower_losses)

    return result_loss


def build_train_MSR_face_graph_multi_gpu(
    batch_size=64,
    num_gpus=4,
    latent_dims=1024,
    lr_g=1e-4,
    lr_c=5e-5,
    clamp_lower=-0.01,
    clamp_upper=0.01
):
    from data_utils import get_msr_train_faces
    with tf.device('/cpu:0'):
        opt_g = tf.train.RMSPropOptimizer(lr_g)
        opt_c = tf.train.RMSPropOptimizer(lr_c)
        real_img , _ = get_msr_train_faces(batch_size,'data/msr_train_happy.tfrecords')
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([real_img], capacity=2 * num_gpus)

        tower_grads_c = []
        tower_grads_g = []
        tower_c_losses = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                image_batch = batch_queue.dequeue()
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)):
                        z=tf.random_normal([batch_size,latent_dims])

                        with tf.variable_scope('generator'):
                            generate_img = generator(z, [4, 4, 128], [64, 64, 3], tf.tanh,
                                                     L.xavier_initializer(uniform=False))
                            tf.get_variable_scope().reuse_variables()


                        fake_logit = critic(generate_img, 32, 4, L.xavier_initializer(uniform=False),
                                            (True if i>=1 else False))
                        true_logit = critic(image_batch, 32, 4, L.xavier_initializer(uniform=False), True)

                        tf.get_variable_scope().reuse_variables()

                        theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                        theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

                        c_loss = tf.reduce_mean(fake_logit - true_logit,name='c_loss')
                        g_loss = tf.reduce_mean(-fake_logit,name='g_loss')

                        tower_c_losses.append(c_loss)

                        tower_grads_c.append(opt_c.compute_gradients(c_loss,var_list=theta_c))
                        tower_grads_g.append(opt_g.compute_gradients(g_loss,var_list=theta_g))

        average_grads_c=average_gradients(tower_grads_c)
        average_grads_g=average_gradients(tower_grads_g)

        total_c_loss=average_loss(tower_c_losses)

        tf.summary.scalar("critic_loss",total_c_loss)

        # for g in average_grads_g:
        #     variable_summaries(g[0],g[0].name)

        tf.summary.image('img', generate_img, max_outputs=10)
        variable_summaries(generate_img,'generated_img')

        apply_gradient_c = opt_c.apply_gradients(average_grads_c)
        apply_gradient_g = opt_g.apply_gradients(average_grads_g)

        theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_c]
        # merge the clip operations on critic variables
        with tf.control_dependencies([apply_gradient_c]):
            apply_gradient_c = tf.tuple(clipped_var_c)


    return apply_gradient_g,apply_gradient_c,total_c_loss

def train_MSR_face_multi_gpu():

    prefix='WGAN_Emotion'
    batch_size=64
    converge_c_iter=100
    converge_g_iter=10
    revise_c_iter=10
    max_iter_step=200000

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement=True

    sess=tf.InteractiveSession()
    opt_g,opt_c,c_loss=build_train_MSR_face_graph_multi_gpu(batch_size=batch_size,latent_dims=2048,
                                                          lr_g=5e-3,lr_c=5e-4,
                                                          clamp_lower=-0.01,clamp_upper=0.01)

    # print([x.name for x in tf.global_variables()])
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tf.global_variables_initializer().run()
    summary_writer = tf.summary.FileWriter('log/'+prefix+'/',sess.graph)


    fc=open('c_loss.log','w')
    fg=open('g_loss.log','w')
    log_interval=50
    for i in range(max_iter_step):
        if i < converge_g_iter or i % 500 == 0:
            citers = converge_c_iter
        else:
            citers = revise_c_iter
            # citers=0

        for j in range(citers):
            if i % log_interval == log_interval-1 and j == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, merged,loss_val = sess.run([opt_c, merged_all,c_loss],
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged)
                summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(i), i)
                print('loss val {}'.format(loss_val))
                fc.write('loss val {}\n'.format(loss_val))
            else:
                sess.run(opt_c)

        # print('optimize g')
        if i % log_interval == log_interval - 1:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, merged,loss_val = sess.run([opt_g, merged_all,c_loss],options=run_options, run_metadata=run_metadata)
            summary_writer.add_summary(merged)
            summary_writer.add_run_metadata(run_metadata, 'generator_metadata {}'.format(i), i)
            print('generator optimize loss val {}'.format(loss_val))
            fg.write('loss val {}\n'.format(loss_val))
        else:
            sess.run(opt_g)

        if i % 1000 == 999:
            saver.save(sess, 'model/'+prefix+'/model.ckpt', global_step=i)

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__=="__main__":
    train_MSR_face_multi_gpu()




