import tensorflow as tf
import cv2
import json
import numpy as np
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_MSR_emotion_face(shape=(128,128),out_file='data/msr_train_happy.tfrecords'):
    with open('data/labels.json','r') as f:
        labels=json.load(f)
    with open('data/paths.json','r') as f:
        paths=json.load(f)
    writer=tf.python_io.TFRecordWriter(out_file)
    count=0
    for index in range(len(paths)):
        if count>20000 or (labels[index]!=9 and labels[index]!=8 and labels[index]!=10 and labels[index]!=1):
            continue
        img_path='d:/data/msr_train_cropped/'+paths[index].split('/')[-1]
        img=cv2.imread(img_path)
        if img.shape[0]<64 or img.shape[1]<64:
            continue
        count+=1
        # print(paths[index])
        # print(img_path)
        # print(img.shape)
        img=cv2.resize(img,shape,cv2.INTER_LINEAR)
        if len(img.shape)==2:
            img=np.expand_dims(img,axis=2)
            img=np.repeat(img,3,axis=2)
        if img.shape[2]==1:
            img = np.repeat(img, 3, axis=2)

        image_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(img.shape[0]),
            'width': _int64_feature(img.shape[1]),
            'depth': _int64_feature(img.shape[2]),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())

    print(count)

    writer.close()

import os

def convert_cartoon_face(shape=(96,96),out_file='data/cartoon_face.tfrecords'):
    writer=tf.python_io.TFRecordWriter(out_file)
    count=0
    for file in os.listdir(r'\\stcvm-216\FaceEmotionData\WU\Image\Face_Valid'):
        img=cv2.imread(r'\\stcvm-216\FaceEmotionData\WU\Image\Face_Valid'+'\\'+file)
        if img.shape[0]<shape[0]:
            continue
        count+=1
        if count%5000==0:
            print(count)

        img=cv2.resize(img,shape,cv2.INTER_LINEAR)
        image_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(img.shape[0]),
            'width': _int64_feature(img.shape[1]),
            'depth': _int64_feature(img.shape[2]),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())

    writer.close()
    print('total {} images'.format(count))

def generate_cartoon_face(num=50000):
    from WGAN import  generator
    writer=tf.python_io.TFRecordWriter('data/cartoon_face_encoder.tfrecords')
    batch_size=500
    restore_epoch=17199

    z = tf.placeholder(tf.float32, [batch_size, 1024])
    with tf.variable_scope('generator'):
        gan = generator(z, [4, 4, 1024], [64, 64, 3], tf.tanh,
                        tf.random_normal_initializer(stddev=0.02), 5, False)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./model/WGAN_Cartoon/model.ckpt-{}".format(restore_epoch))

        for _ in range(int(num/batch_size)):
            batch_z = np.random.normal(0, 1.0, [batch_size, 1024]).astype(np.float32)
            rs = gan.eval(feed_dict={z: batch_z})
            rs=(rs+1.0)/2.0*255
            for i in range(batch_size):
                img=np.array(rs[i],dtype=np.uint8)
                img_latent=np.array(batch_z[i],dtype=np.float32).flatten()#.tostring()
                image_raw = img.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    # 'height': _int64_feature(img.shape[0]),
                    # 'width': _int64_feature(img.shape[1]),
                    # 'depth': _int64_feature(img.shape[2]),
                    'image_raw': _bytes_feature(image_raw),
                    'image_latent':_float_feature(img_latent)}))
                writer.write(example.SerializeToString())

    writer.close()
if __name__=="__main__":
    convert_cartoon_face((128,128),'data/user_face.tfrecords')