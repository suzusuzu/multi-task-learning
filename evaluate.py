# coding: utf-8

import os
import cv2
import numpy as np
import tensorflow as tf
import BKNetStyle2 as BKNetStyle
from const import *
from mtcnn.mtcnn import MTCNN


def load_network():
    sess = tf.Session()
    x = tf.placeholder(tf.float32, [None, 48, 48, 1])
    y_smile_conv, y_gender_conv, y_age_conv, phase_train, keep_prob = BKNetStyle.BKNetModel(x)
    saver = tf.train.Saver()
    saver.restore(sess, './multi-task-learning/save/current5/model-age101.ckpt.index')
    return sess, x, y_smile_conv, y_gender_conv, y_age_conv, phase_train, keep_prob


'''
def draw_label(image, x, y, w, h, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    cv2.rectangle(image, (x, y), (x+w, y+h), (0,155,255), 2)
    cv2.putText(image, label, (x,y), font, font_scale, (255, 255, 255), thickness)
'''

def main(sess, x, y_smile_conv, y_gender_conv, y_age_conv, phase_train, keep_prob, inp):
    detector = MTCNN()
    img = cv2.imread(inp)

    # detect face and crop face, convert to gray, resize to 48x48
    original_img = img
    result = detector.detect_faces(original_img)
    if result:
        face_position = result[0].get('box')
        x_coordinate = face_position[0]
        y_coordinate = face_position[1]
        w_coordinate = face_position[2]
        h_coordinate = face_position[3]
        img = original_img[y_coordinate:y_coordinate+h_coordinate, x_coordinate:x_coordinate+w_coordinate]
        if(img.size==0):
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(48, 48))
        img = (img - 128) / 255.0
        T = np.zeros([48, 48, 1])
        T[:, :, 0] = img
        test_img = []
        test_img.append(T)
        test_img = np.asarray(test_img)
        predict_y_smile_conv = sess.run(y_smile_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
        print(predict_y_smile_conv[0][1])



if __name__ == '__main__':
    sess, x, y_smile_conv, y_gender_conv, y_age_conv, phase_train, keep_prob = load_network()
    while True:
        inp=input()
        main(sess, x, y_smile_conv, y_gender_conv, y_age_conv, phase_train, keep_prob, inp)
