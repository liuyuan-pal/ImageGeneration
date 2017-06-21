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
import cv2
import json
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def convert_MSR_emotion_face(shape=(128,128),out_file='data/msr_train.tfrecords'):
    with open('data/labels.json','r') as f:
        labels=json.load(f)
    with open('data/paths.json','r') as f:
        paths=json.load(f)
    writer=tf.python_io.TFRecordWriter(out_file)
    for index in range(len(paths)):
        img_path='d:/data/msr_train_cropped/'+paths[index].split('/')[-1]
        img=cv2.imread(img_path)
        # print(paths[index])
        # print(img_path)
        # print(img.shape)
        img=cv2.resize(img,shape)
        image_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(img.shape[0]),
            'width': _int64_feature(img.shape[1]),
            'depth': _int64_feature(img.shape[2]),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

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
    image= tf.reshape(image,[shape[0],shape[1],shape[2]])
    label = tf.cast(features['label'], tf.int32)

    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=batch_size,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)


    return images, sparse_labels

if __name__=="__main__":
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
            img=cv2.cvtColor(example[i],cv2.COLOR_BGR2RGB)
            print(label)
            plt.imshow(img)
            plt.show()
