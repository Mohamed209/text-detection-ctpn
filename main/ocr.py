# coding=utf-8
import os
import shutil
import sys
import time
import math
import cv2
import numpy as np
import tensorflow as tf
import pyarabic.araby as araby
import string
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import keras.backend as K
from keras.models import load_model
from collections import Counter
from string import punctuation
###############################################################
import random
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import interpolation as inter
from typing import Tuple, Union
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
from skimage import data
import skimage.filters as filters
##################################################################
sys.path.append(os.getcwd())
from utils.text_connector.detectors import TextDetector
from utils.rpn_msr.proposal_layer import proposal_layer
from nets import model_train as model
tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('ocr_output_path', 'data/ocr_res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG', 'tiff']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def show_img(img, title="test"):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ocrline(line, model, letters):
    line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    line = cv2.resize(line, (432, 32))
    line = line/255.0
    line = np.expand_dims(line, -1)
    line = np.expand_dims(line, axis=0)
    prediction = model.predict(line)
    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                                   greedy=True)[0][0])
    # see the results
    i = 0
    text = ''
    for x in out:
        print("predicted text = ", end='')
        for p in x:
            if int(p) != -1:
                try:
                    print(letters[int(p)], end='')
                    text += letters[int(p)]
                except IndexError:
                    pass
        print('\n')
        i += 1
    return text


def supress_ocr_noise(text, special_chars=list(set(punctuation))):
    '''
    handle very long repeated chars , by supressing it
    '''
    text_len = len(text)
    freq = Counter(text)
    for key in freq:
        if freq[key] > int(0.50*text_len) and key in special_chars:
            print('suppressed ', text)
            return ''  # more than 55% of string are one char
    return text


def sort_boxes(pts):
    return np.array(sorted(pts, key=lambda k: [k[1], k[0]]))


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def main(argv=None):
    if os.path.exists(FLAGS.ocr_output_path):
        shutil.rmtree(FLAGS.ocr_output_path)
    os.makedirs(FLAGS.ocr_output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(
            tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(
            0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(
                ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            letters = araby.LETTERS+string.printable+u'٠١٢٣٤٥٦٧٨٩'
            ocr_arch_path = 'nets/ocr/ocr_model.h5'
            ocr_weights_path = 'checkpoints_mlt/ocr/afev_slimCRNN--10--0.614.hdf5'
            ocr = load_model(ocr_arch_path)
            ocr.load_weights(ocr_weights_path)
            for im_fn in im_fn_list:
                img_name = os.path.basename(im_fn)
                print('===============')
                print(im_fn)
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue
                img, (rh, rw) = resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                textsegs, _ = proposal_layer(
                    cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='O')
                boxes = textdetector.detect(
                    textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)
                boxes = sort_boxes(boxes)
                cost_time = (time.time() - start)
                print("cost time: {:.2f}s".format(cost_time))
                for idx, box in enumerate(boxes):
                    points = box[:8].astype(np.int32).reshape(4, 2)
                    line = four_point_transform(img,points)
                    #show_img(line,'line')
                    with open(FLAGS.ocr_output_path+im_fn.split('/')[2].split('.')[0]+'_ocr.txt', mode='a+', encoding='utf-8') as res:
                        prediction = ocrline(line, ocr, letters)
                        prediction = supress_ocr_noise(prediction)
                        try:
                            res.writelines(prediction)
                        except TypeError:
                            pass
                        res.write('\n')


if __name__ == '__main__':
    tf.app.run()
