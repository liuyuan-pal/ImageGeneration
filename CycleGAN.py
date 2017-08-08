import tensorflow as tf
import tensorflow.contrib.layers as layers
from WGAN import lrelu

def residual_generator_block(h,batch_para,num_filters=128):
    m=layers.conv2d(h,num_filters,3,1,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=batch_para)

    m=layers.conv2d(m,num_filters,3,1,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),)
    # residual
    m=m+h

    m=layers.batch_norm(m,**batch_para)
    m=tf.nn.relu(m)

    return m

def generator(input_img,for_train):

    batch_para={'is_training':for_train,'center':True,'scale':True}

    # c7s1-32
    h=layers.conv2d(input_img,32,7,1,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=batch_para)

    # d64
    h=layers.conv2d(h,64,3,2,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=batch_para)
    # d128
    h=layers.conv2d(h,128,3,2,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=batch_para)

    # R128
    for i in range(6):
        h=residual_generator_block(h,batch_para,128)

    # u64
    h=layers.conv2d_transpose(h,64,3,2,
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                              activation_fn=tf.nn.relu,
                              normalizer_fn=layers.batch_norm,
                              normalizer_params=batch_para)

    # u32
    h=layers.conv2d_transpose(h,32,3,2,
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                              activation_fn=tf.nn.relu,
                              normalizer_fn=layers.batch_norm,
                              normalizer_params=batch_para)

    # c7s1-3
    output_img=layers.conv2d(h,3,7,1,
                             weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                             activation_fn=tf.tanh)

    return output_img

def discriminator(input_img,for_train):
    batch_para={'is_training':for_train,'center':True,'scale':True}
    # c64
    h=layers.conv2d(input_img,64,5,2,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    activation_fn=lrelu,
                    normalizer_fn=None)
    # c128
    h=layers.conv2d(h,128,5,2,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    activation_fn=lrelu,)
                    # normalizer_fn=layers.batch_norm,
                    # normalizer_params=batch_para)
    # c256
    h=layers.conv2d(h,256,5,2,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    activation_fn=lrelu,)
                    # normalizer_fn=layers.batch_norm,
                    # normalizer_params=batch_para)
    # c512
    h=layers.conv2d(h,512,5,2,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    activation_fn=lrelu,)
                    # normalizer_fn=layers.batch_norm,
                    # normalizer_params=batch_para)

    # output
    h=layers.flatten(h)
    logits=layers.fully_connected(h,1,None,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.02),)

    return logits

def build_graph(batch_size,lr,num_gpus,grad_norm=10,recon_scale=10):
    from data_utils import get_cartoon_faces
    from WGAN import average_loss,average_gradients
    with tf.device('/cpu:0'):
        opt_g = tf.train.AdamOptimizer(lr)
        opt_c = tf.train.AdamOptimizer(lr)

        real_img=get_cartoon_faces(batch_size,'data/user_face.tfrecords')
        real_batch = tf.contrib.slim.prefetch_queue.prefetch_queue([real_img], capacity=2 * num_gpus)

        cart_img=get_cartoon_faces(batch_size,'data/cartoon_face.tfrecords')
        cart_batch = tf.contrib.slim.prefetch_queue.prefetch_queue([cart_img], capacity=2 * num_gpus)


        c_grads=[]
        g_grads=[]
        c_losses=[]

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                real = real_batch.dequeue()
                cart = cart_batch.dequeue()
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)):


                        with tf.variable_scope('generator'):
                            with tf.variable_scope('face2cartoon'):
                                fake_cart=generator(real,True)
                            with tf.variable_scope('cartoon2face'):
                                fake_real=generator(cart,True)

                            alpha_cart = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
                            alpha_face = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)

                            inte_cart = fake_cart * alpha_cart + (1.0 - alpha_cart) * cart
                            inte_real = fake_real * alpha_face + (1.0 - alpha_face) * real

                            tf.get_variable_scope().reuse_variables()

                            with tf.variable_scope('face2cartoon'):
                                recon_cart=generator(fake_real,True)
                            with tf.variable_scope('cartoon2face'):
                                recon_real=generator(fake_cart,True)

                        with tf.variable_scope('critic'):
                            with tf.variable_scope('cart'):
                                fake_cart_logit=discriminator(fake_cart,True)
                                true_cart_logit=discriminator(cart,True)
                                inte_cart_logit=discriminator(inte_cart,True)

                            with tf.variable_scope('face'):
                                fake_real_logit=discriminator(fake_real,True)
                                true_real_logit=discriminator(real,True)
                                inte_real_logit=discriminator(inte_real,True)

                        tf.get_variable_scope().reuse_variables()

                        c_loss=tf.reduce_mean(fake_real_logit) - tf.reduce_mean(true_real_logit),\
                               +tf.reduce_mean(fake_cart_logit) - tf.reduce_mean(true_cart_logit)

                        g_loss=tf.reduce_mean(-fake_real_logit,name='g_real_loss')+\
                               tf.reduce_mean(-fake_cart_logit,name='g_cart_loss')

                        recon_loss=tf.reduce_mean(tf.reduce_mean(tf.abs(real - recon_real),axis=[1,2,3]))\
                                  +tf.reduce_mean(tf.reduce_mean(tf.abs(cart - recon_cart),axis=[1,2,3]))


                        cart_grad=tf.gradients(inte_cart_logit,inte_cart)[0]
                        cart_ddx=tf.sqrt(tf.reduce_sum(tf.square(cart_grad),axis=[1,2,3]))
                        cart_ddx=tf.reduce_mean(tf.square(cart_ddx-1.0))

                        real_grad=tf.gradients(inte_real_logit,inte_real)[0]
                        real_ddx=tf.sqrt(tf.reduce_sum(tf.square(real_grad),axis=[1,2,3]))
                        real_ddx=tf.reduce_mean(tf.square(real_ddx-1.0))

                        c_loss=c_loss+grad_norm*(cart_ddx+real_ddx)

                        theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                        theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

                        c_grad=opt_c.compute_gradients(c_loss,var_list=theta_c)
                        c_grads.append(c_grad)

                        total_g_loss=g_loss+recon_scale*recon_loss
                        g_grad=opt_g.compute_gradients(total_g_loss,var_list=theta_g)
                        g_grads.append(g_grad)
                        # print(g_grad)

                        c_losses.append(c_loss)

        average_grads_c=average_gradients(c_grads)
        average_grads_g=average_gradients(g_grads)
        total_loss=average_loss(c_losses)

        tf.summary.scalar('total critic loss',total_loss)
        tf.summary.image('fake face',fake_real,10)
        tf.summary.image('fake cartoon', fake_cart, 10)
        tf.summary.image('reconstruct face', recon_real, 10)
        tf.summary.image('reconstruct cartoon', recon_cart, 10)


        apply_gradient_c = opt_c.apply_gradients(average_grads_c)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # add dependency of updating moving statistics of batch normalization
        with tf.control_dependencies(update_ops):
            apply_gradient_g = opt_g.apply_gradients(average_grads_g)

        return apply_gradient_g,apply_gradient_c,total_loss


def train_cycle_gan():

    prefix='CycleGAN'
    batch_size=16
    converge_c_iter=300
    converge_g_iter=25
    revise_c_iter=30
    max_iter_step=200000
    # restore_epoch=9099
    learning_rate=1e-5

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement=True

    sess=tf.InteractiveSession()
    opt_g,opt_c,c_loss=build_graph(batch_size,learning_rate,4)

    trainable_var=[x.name for x in tf.global_variables()]
    for var in trainable_var:
        print(var)

    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tf.global_variables_initializer().run()
    # saver.restore(sess,"model/"+prefix+"/model.ckpt-{}".format(restore_epoch))
    summary_writer = tf.summary.FileWriter('log/'+prefix+'/',sess.graph)

    log_interval=5
    save_interval=100
    try:
        for i in range(max_iter_step):
            if i < converge_g_iter or i % 50 == 0:
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
                else:
                    sess.run(opt_c)

            # print('optimize g')
            if i % log_interval == log_interval - 1:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, merged,loss_val = sess.run([opt_g, merged_all,c_loss],
                                              options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged)
                summary_writer.add_run_metadata(run_metadata, 'generator_metadata {}'.format(i), i)
                print('generator optimize loss val {}'.format(loss_val))
            else:
                sess.run(opt_g)

            if i % save_interval == save_interval-1:
                saver.save(sess, 'model/'+prefix+'/model.ckpt', global_step=i)
    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()

def test_part():
    batch_size=30
    from data_utils import get_cartoon_faces
    real_img = get_cartoon_faces(batch_size, 'data/user_face.tfrecords')
    real_batch = tf.contrib.slim.prefetch_queue.prefetch_queue([real_img], capacity=2 * 4)

    cart_img = get_cartoon_faces(batch_size, 'data/cartoon_face.tfrecords')
    cart_batch = tf.contrib.slim.prefetch_queue.prefetch_queue([cart_img], capacity=2 * 4)

    real = real_batch.dequeue()
    cart = cart_batch.dequeue()

    with tf.variable_scope('generator'):
        with tf.variable_scope('face2cartoon'):
            fake_cart = generator(real, True)
        with tf.variable_scope('cartoon2face'):
            fake_real = generator(cart, True)

        tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('face2cartoon'):
            recon_cart = generator(fake_real, True)
        with tf.variable_scope('cartoon2face'):
            recon_real = generator(fake_cart, True)

    sess=tf.InteractiveSession()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    tf.global_variables_initializer().run()

    fake_real_img,fake_cart_img = sess.run([recon_cart,recon_real])

    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(batch_size):
        img1=np.asarray(fake_real_img[i]).flatten()
        img2=np.asarray(fake_cart_img[i]).flatten()
        print(min(img1),max(img1),min(img2),max(img2))
        # img1=np.array((real_img[i]+1.0)/2.0*255,dtype=np.uint8)
        # img2=np.array((cart_img[i]+1.0)/2.0*255,dtype=np.uint8)
        plt.subplot(211)
        plt.hist(img1)
        plt.subplot(212)
        plt.hist(img2)
        plt.show()



if __name__=="__main__":
    test_part()


