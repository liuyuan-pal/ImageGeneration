import tensorflow as tf
import cv2
import json
import numpy as np
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
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

if __name__=="__main__":
    convert_MSR_emotion_face()