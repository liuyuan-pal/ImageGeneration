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

if __name__=="__main__":
    import cv2
    images,labels=get_msr_train_faces(20)

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Retrieve a single instance:
        example, label = sess.run([images, labels])
        import matplotlib.pyplot as plt
        for i in range(20):
            print(example[i].shape)
            plt.figure()
            plt.hist(example[i].ravel())
            plt.figure()
            img=example[i]*128+128
            img=np.asarray(img,np.uint8)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            print(label)
            plt.imshow(img)
            plt.show()
