import tensorflow as tf


def encoder(img,for_train=True,channel_nums=(64,128,256,512)):
    with tf.variable_scope('encoder'):
        for i in range(len(channel_nums)):
            img = tf.contrib.layers.conv2d(img, channel_nums[i], 5, 2,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.nn.relu,
                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                           normalizer_params={'is_training': for_train,
                                                             'center':True,'scale':True})

        fc = tf.contrib.layers.flatten(img)
        fc1 = tf.contrib.layers.fully_connected(fc,1024,weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                activation_fn=tf.nn.relu,normalizer_fn=tf.contrib.layers.batch_norm,
                                                normalizer_params={'is_training': for_train,'center':True,'scale':True})
        latent = tf.contrib.layers.fully_connected(fc1,1024,weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   activation_fn=None,normalizer_fn=None)

        return latent

from WGAN import average_loss,average_gradients

def build_train_graph(
        lr,
        batch_size,
        num_gpus=4,

):
    from data_utils import get_cartoon_faces_auto_encoder
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr=tf.train.exponential_decay(lr,global_step,1000,0.9)
        opt = tf.train.AdamOptimizer(lr)


        img,target = get_cartoon_faces_auto_encoder(batch_size)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([img,target], capacity=2 * num_gpus)

        valid_img,valid_target = get_cartoon_faces_auto_encoder(batch_size,'data/cartoon_face_encoder_valid.tfrecords')
        valid_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([valid_img,valid_target], capacity=2 * num_gpus)

        tower_grads=[]
        tower_losses=[]
        tower_valid_losses=[]
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                image_batch,target_batch = batch_queue.dequeue()
                valid_img_batch,valid_tgt_batch = valid_batch_queue.dequeue()
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)):
                        latent_val=encoder(image_batch)
                        tf.get_variable_scope().reuse_variables()
                        valid_latent_val=encoder(valid_img_batch)

                        loss=tf.reduce_mean(tf.reduce_sum(tf.pow(latent_val - target_batch, 2),axis=1),axis=0)
                        valid_loss=tf.reduce_mean(tf.reduce_sum(tf.pow(valid_latent_val - valid_tgt_batch, 2),axis=1),axis=0)

                        gradient=opt.compute_gradients(loss)

                        tower_losses.append(loss)
                        tower_grads.append(gradient)
                        tower_valid_losses.append(valid_loss)

        grads=average_gradients(tower_grads)
        total_loss=average_loss(tower_losses)
        valid_total_loss=average_loss(tower_valid_losses)

        tf.summary.scalar('train loss',total_loss)
        tf.summary.scalar('valid loss',valid_total_loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # add dependency of updating moving statistics of batch normalization
        with tf.control_dependencies(update_ops):
            apply_grad=opt.apply_gradients(grads)

        return apply_grad,total_loss,valid_total_loss


def train_encoder():
    save_interval=200
    summary_interval=20
    train_steps=500000
    prefix='encoder'
    # restore_epoch=12599

    with tf.Session() as sess:
        opt,loss,valid_loss=build_train_graph(1e-4,32,4)

        print([x.name for x in tf.global_variables()])
        merged_all = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=10)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.global_variables_initializer().run()
        # saver.restore(sess,"model/"+prefix+"/model.ckpt-{}".format(restore_epoch))
        summary_writer = tf.summary.FileWriter('log/' + prefix + '/', sess.graph)


        for i in range(train_steps):
            if i%summary_interval==summary_interval-1:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, merged,loss_val, valid_loss_val = sess.run([opt, merged_all, loss, valid_loss],
                                              options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged)
                summary_writer.add_run_metadata(run_metadata, 'loss summary {}'.format(i), i)
                print('loss val {}'.format(loss_val))
                print('valid loss val {}'.format(valid_loss_val))
            else:
                sess.run(opt)

            if i % save_interval==save_interval-1:
                saver.save(sess,'model/'+prefix+'/model.ckpt',global_step=i)


    coord.request_stop()
    coord.join(threads)
    sess.close()


import cv2
def build_encoder_generator(batch_imgs):
    from WGAN import generator
    img=tf.placeholder(tf.float32,[None,64,64,3])
    z=encoder(img,False)
    # z=z/tf.reduce_max(tf.abs(z),axis=1)/100000
    z_mean=tf.reduce_mean(z,axis=1)
    z_stddev=tf.sqrt(tf.reduce_sum(tf.pow((z-z_mean),2),axis=1))
    z-=z_mean
    z/=z_stddev*0.025
    # print(z.shape)
    # z=tf.clip_by_value(z,-1,1)
    # z=tf.random_normal([1,1024],stddev=1.0)
    with tf.variable_scope("generator"):
        generated_img=generator(z,[4, 4, 1024], [64, 64, 3],tf.tanh,for_train=False)

    with tf.Session() as sess:
        generator_saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
        generator_saver.restore(sess,'model/WGAN_Cartoon/model.ckpt-17199')
        encoder_saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'))
        encoder_saver.restore(sess,'model/encoder/model.ckpt-31799')

        result_img,z_val=sess.run([generated_img,z],{img:batch_imgs})
        print(min(z_val[0]),max(z_val[0]))
        # z_val=np.array(z_val)
        # print(z_val.shape)
        # print(np.sum(z_val[0]**2),np.mean(z_val[0]))
        for index,rimg in enumerate(result_img):
            oimg=((rimg+1.0)/2.0*255)
            cv2.imwrite('result/trans/{}.jpg'.format(index),oimg)

        import matplotlib.pyplot as plt
        val=np.random.normal(size=[1,1024])
        plt.subplot(211)
        plt.hist(val[0])
        plt.subplot(212)
        plt.hist(z_val[0])
        plt.show()

if __name__=="__main__":
    import numpy as np
    img=cv2.imread(r'D:\projects\DeepGenerativeModel\result\trans\contempt_contempt_0bbaabb0-29b2-11e7-ac4d-10604b88873d.jpg')
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_LINEAR)
    img=np.expand_dims(img,axis=0)
    img=img.astype(np.float32)
    img=(img-128.0)/129.0
    build_encoder_generator(img)