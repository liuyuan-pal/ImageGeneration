import numpy as np

def load_MNIST():
    with open('data/train-images.idx3-ubyte','rb') as f:
        f.seek(16)
        buffer=f.read()
        X=np.frombuffer(buffer,dtype=np.uint8)
    # print X.shape
    X=X.reshape([60000,28,28])
    return X

import tensorflow as tf
def get_msr_train_faces(batch_size,filename='data/msr_train.tfrecords',shape=(128,128,3)):
    reader=tf.TFRecordReader()
    filename_queue=tf.train.string_input_producer([filename])
    _,serialized_example=reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image,[shape[0],shape[1],shape[2]])
    image = tf.cast(image, tf.float32)
    # reshape and normalize
    image = tf.image.resize_images(image,tf.constant([64,64],tf.int32))
    image = (image-tf.constant(128,tf.float32))/tf.constant(128,tf.float32)

    label = tf.cast(features['label'], tf.int32)

    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=batch_size,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)

    return images, sparse_labels


def get_cartoon_faces(batch_size,filename='data/cartoon_face.tfrecords',shape=(96,96,3)):
    reader=tf.TFRecordReader()
    filename_queue=tf.train.string_input_producer([filename])
    _,serialized_example=reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image,[shape[0],shape[1],shape[2]])
    image = tf.cast(image, tf.float32)
    # reshape and normalize
    image = tf.image.resize_images(image,tf.constant([64,64],tf.int32))
    image = (image-tf.constant(128,tf.float32))/tf.constant(129,tf.float32)

    # label = tf.cast(features['label'], tf.int32)

    images = tf.train.shuffle_batch(
        [image], batch_size=batch_size, num_threads=batch_size,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)

    return images

def get_cartoon_faces_auto_encoder(batch_size,filename='data/cartoon_face_encoder.tfrecords',shape=(64,64,3)):
    reader=tf.TFRecordReader()
    filename_queue=tf.train.string_input_producer([filename])
    _,serialized_example=reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'image_latent': tf.FixedLenFeature([1024], tf.float32),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image,[shape[0],shape[1],shape[2]])
    image = tf.cast(image, tf.float32)
    # reshape and normalize
    image = tf.image.resize_images(image,tf.constant([64,64],tf.int32))
    image = (image-tf.constant(128,tf.float32))/tf.constant(129,tf.float32)

    label = tf.cast(features['image_latent'], tf.float32)
    print(label.shape)
    # label = tf.reshape(label,1024)

    images,labels = tf.train.shuffle_batch(
        [image,label], batch_size=batch_size, num_threads=batch_size,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)

    return images,labels

if __name__=="__main__":
    import cv2
    images,labels=get_cartoon_faces_auto_encoder(20)
    print(images)

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Retrieve a single instance:
        example,label = sess.run([images,labels])
        print(len(example))
        import matplotlib.pyplot as plt
        for i in range(20):
            print(example[i].shape)
            print(np.array(label[i]))
            plt.figure()
            plt.hist(example[i].ravel())
            plt.figure()
            img=example[i]*129+128
            img=np.asarray(img,np.uint8)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()
