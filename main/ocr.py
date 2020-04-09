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
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
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


def deskew_image(img, boxes):
    angle_acc = 0
    for i, box in enumerate(boxes):
        pts = box[:8].astype(np.int32).reshape((-1, 1, 2))
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        angle_acc += angle
    angle_acc /= len(boxes)

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_acc, 1.0)
    try:
        img = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except:
        pass

    return img


def crop_image(img, boxes, write_image=True, verbose=False):

    # Sort boxes ascending () Topmost point then Leftmost point )
    boxes = np.array(sorted(boxes, key=lambda k: [k[1], k[0]]))

    # Extract interset points to crop receipt
    # max(0,number) to avoid -1 returning
    leftmost = max(0, min([min(boxes[:, 0]), min(boxes[:, 6])]))
    rightmost = max([max(boxes[:, 2]), max(boxes[:, 4])])
    # max(0,number) to avoid -1 returning
    topmost = max(0, min([min(boxes[:, 1]), min(boxes[:, 3])]))
    bottommost = max([max(boxes[:, 5]), max(boxes[:, 7])])

    # Reshape interset points to the following shape [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    pts = np.array([leftmost, topmost, rightmost, topmost, rightmost, bottommost, leftmost, bottommost])\
        .astype(np.int32).reshape((-1, 2))

    # Create the receipt bounding rectangle from interset points
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = img[y:y+h, x:x+w]

    return cropped, pts

    if write_image:
        cv2.imwrite(os.path.join(FLAGS.output_path, "cropped_" +
                                 img_name.replace('jpeg', 'tiff').replace('jpg', 'tiff')), cropped[:, :, ::-1])


def detect_text(img, sess, bbox_pred, cls_pred, cls_prob, input_image, input_im_info, mode='O'):
    start = time.time()

    h, w, c = img.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                           feed_dict={input_image: [img],
                                                      input_im_info: im_info})
    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
    scores = textsegs[:, 0]
    textsegs = textsegs[:, 1:5]
    textdetector = TextDetector(DETECT_MODE=mode)
    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
    boxes = np.array(boxes, dtype=np.int)
    cost_time = (time.time() - start)
    print("cost time: {:.2f}s".format(cost_time))
    return img, boxes


def merge_boxes(boxes):

    # img = orig_image.copy()

    # Extract interset points to crop receipt
    # max(0,number) to avoid -1 returning
    leftmost = max(0, min([min(boxes[:, 0]), min(boxes[:, 6])]))
    rightmost = max([max(boxes[:, 2]), max(boxes[:, 4])])
    # max(0,number) to avoid -1 returning
    topmost = max(0, min([min(boxes[:, 1]), min(boxes[:, 3])]))
    bottommost = max([max(boxes[:, 5]), max(boxes[:, 7])])

    threshold = 15
    merge_count = 0
    black_list = []
    new_boxes = []
    for i, box in enumerate(boxes):
        # Skip the merged boxes
        if i in black_list:
            continue
        pts = box[:8].astype(np.int32).reshape((-1, 1, 2))
        # cv2.polylines(img, [pts], True, color=(0, 0, 255), thickness=2)
        # show_img(img)

        # Loop on all boxes after current box
        for idx in range(i+1, len(boxes)):
            # Skip the merged boxes
            if idx in black_list:
                continue
            # Set temp_box as the next box
            tmp_box = boxes[idx]
            # Check if Height difference - of one of two corners - less than threshold (i.e the same line)
            if abs(tmp_box[1] - box[1]) < threshold or abs(tmp_box[3] - box[3]) < threshold:
                black_list.append(idx)
                # count how many boxes are merged
                merge_count = merge_count + 1

                # stretch the original width box to cover the two boxes (Consider stretching from LTR or RTL)
                if box[0] >= tmp_box[2]:
                    box[0] = tmp_box[0]
                    box[6] = tmp_box[6]
                elif box[2] <= tmp_box[0]:
                    box[2] = tmp_box[2]
                    box[4] = tmp_box[4]
                # selecet the largest height and set the original box to the larger one (to avoid clipping)
                max_height_left_corner = np.min(
                    [box[1], box[3], tmp_box[1], tmp_box[3]])
                box[1] = box[3] = max_height_left_corner
                # selecet the largest lower height and set the original box to the larger one (to avoid clipping)
                max_height_right_corner = np.max(
                    [box[5], box[7], tmp_box[5], tmp_box[7]])
                box[5] = box[7] = max_height_right_corner

        box[0] = box[6] = leftmost
        box[2] = box[4] = rightmost
        new_boxes.append(box)
        pts = box[:8].astype(np.int32).reshape((-1, 1, 2))
    new_boxes = np.array(sorted(new_boxes, key=lambda k: [k[1], k[0]]))
    return new_boxes


def sub_line_equation(x1, y1, x2, y2, x=None, y=None):
    m = (y1 - y2) / (x1 - x2)
    if y is None:
        y_calc = m * (x - x1) + y1
        return y_calc
    elif x is None:
        x_calc = ((y - y1) / m) + x1
        return x_calc

    return (x_calc + x, y_calc + y)


def get_relative_distance(orig_pts, boxes):
    line1 = np.reshape([orig_pts[0], orig_pts[1]], -1)
    line2 = np.reshape([orig_pts[0], orig_pts[3]], -1)
    for idx, box in enumerate(boxes):
        box = box[:8].astype(np.int32).reshape((-1, 2))
        for i in range(0, 8, 2):
            boxes[idx][i] = boxes[idx][i] + \
                sub_line_equation(line2[0], line2[1],
                                  line2[2], line2[3], y=boxes[idx][i+1])
            boxes[idx][i+1] = boxes[idx][i+1] + \
                sub_line_equation(line1[0], line1[1],
                                  line1[2], line1[3], x=boxes[idx][i])
    return boxes


def crop_boxes(img, boxes):
    lines = []
    for i, box in enumerate(boxes):

        pts = box[:8].astype(np.int32).reshape((-1, 1, 2))
        pts[pts < 0] = 0
        line_rect = cv2.boundingRect(pts)
        x, y, w, h = line_rect
        croped_line = img[y:y+h, x:x+w].copy()
        #croped_line = deskew_image(croped_line, [box])
        #croped_line = cv2.cvtColor(croped_line, cv2.COLOR_BGR2GRAY)
        #_,croped_line = cv2.threshold(croped_line, 0,255,cv2.THRESH_OTSU)
        #croped_line = cv2.cvtColor(croped_line,cv2.COLOR_GRAY2RGB)
        lines.append(croped_line)
    return lines


def stretch_boxes(input_img_shape, resized_image_shape, boxes):
    input_w = input_img_shape[0]
    input_h = input_img_shape[1]
    resized_w = resized_image_shape[0]
    resized_h = resized_image_shape[1]
    ratio_w = (input_w / resized_w)
    ratio_h = (input_h / resized_h)
    for box in boxes:
        box[0] *= ratio_w
        box[2] *= ratio_w
        box[4] *= ratio_w
        box[6] *= ratio_w
        box[1] *= ratio_h
        box[3] *= ratio_h
        box[5] *= ratio_h
        box[7] *= ratio_h
    return boxes


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    print("image skewed with {} degrees".format(best_angle))
    return rotated


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


def supress_noise(text, special_chars=list(set(punctuation))):
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


def main(argv=None):
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
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
            arch_path = 'nets/ocr/ocr_model.h5'
            weights_path = 'checkpoints_mlt/ocr/afev_slimCRNN--10--0.614.hdf5'
            ocr = load_model(arch_path)
            ocr.load_weights(weights_path)
            for im_fn in im_fn_list:
                img_name = os.path.basename(im_fn)
                print('===============')
                print(im_fn)
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue
                im = correct_skew(im)
                input_img = im
                # Resize image to be feed to the network
                resized_img, (rh, rw) = resize_image(input_img)
                # show_img(resized_img,title='original')
                # Detect text from the resized image
                img, boxes = detect_text(
                    resized_img, sess, bbox_pred, cls_pred, cls_prob, input_image, input_im_info, mode='O')
                # Rescale box size (in order to plot it on the orignal image not the resized version)
                boxes = stretch_boxes(
                    input_img.shape, resized_img.shape, boxes)
                # crop receipt
                img, orig_pts = crop_image(input_img, boxes, False)
                img = correct_skew(img)
                #show_img(img, 'cropped')
                # Detect text again from the cropped image
                img, (rh, rw) = resize_image(img)
                img, boxes = detect_text(
                    img, sess, bbox_pred, cls_pred, cls_prob, input_image, input_im_info, mode='O')
                mergboxes = merge_boxes(boxes)
                lines = crop_boxes(img, mergboxes)
                for idx, line in enumerate(lines):
                    # ocr the line
                    line = correct_skew(line)
                    #show_img(line, title='line'+str(idx))
                    with open(FLAGS.output_path+im_fn.split('/')[2].split('.')[0]+'_ocr.txt', mode='a+', encoding='utf-8') as res:
                        prediction = ocrline(line, ocr, letters)
                        prediction = supress_noise(prediction)
                        try:
                            res.writelines(prediction)
                        except TypeError:
                            pass
                        res.write('\n')


if __name__ == '__main__':
    tf.app.run()
