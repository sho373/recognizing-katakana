import cv2
import time
import math
import os
import argparse
import numpy as np
import tensorflow as tf
from keras.models import load_model, model_from_json
from google_trans_new import google_translator  
from PIL import ImageFont, ImageDraw, Image
from data_generation.for_recognition.katakana_japanese import label_dic

import detection.locality_aware_nms as nms_locality
import detection.lanms

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, default='check/')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--model_path', type=str, default='model/')
parser.add_argument('--output_dir', type=str, default='results/')
FLAGS = parser.parse_args()

from detection.model import *
from detection.losses import *
from detection.data_processor import restore_rectangle

def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR

def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL

def cv2_putText_1(img, text, org, fontFace, color, shape):
	b, g, r = color
	x, y = org
	colorRGB = (r, g, b)
	imgPIL = cv2pil(img)
	draw = ImageDraw.Draw(imgPIL)
	#fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
	draw.rectangle(shape, outline = (0,255,0), width = 2)
	draw.text(xy = (x,y), text = text, fill = colorRGB, font = fontFace)
	imgCV = pil2cv(imgPIL)
	return imgCV

def predict_katakana(image, model):
	
	img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	img = 255 - img
	img = img.reshape(1, 48, 48, 1).astype('float32') / 255
	ret = model.predict(img)
	return ret

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    font = ImageFont.truetype("./fonts/MSGOTHIC.TTC", 20)
    key_list = list(label_dic.keys())
    translator = google_translator()
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    # load trained model
    json_file = open(os.path.join('/'.join(FLAGS.model_path.split('/')[0:-1]), 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': RESIZE_FACTOR})
    model.load_weights(FLAGS.model_path)
    model_recog = load_model('katakana_char_model.h5')

    img_list = get_images()
    for img_file in img_list:
        
        img = cv2.imread(img_file)[:, :, ::-1]
        translated_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start_time = time.time()
        img_resized, (ratio_h, ratio_w) = resize_image(img)

        img_resized = (img_resized / 127.5) - 1

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()

        # feed image into model
        score_map, geo_map = model.predict(img_resized[np.newaxis, :, :, :])

        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score_map, geo_map=geo_map, timer=timer)
        print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            img_file, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        print('[timing] {}'.format(duration))

        # save to file
        if boxes is not None:
            res_file = os.path.join(
                FLAGS.output_dir,
                '{}.txt'.format(
                    os.path.basename(img_file).split('.')[0]))

            with open(res_file, 'w') as f:
                for box in boxes:
                    results_text = ''
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    cropped_image = img[int(box[0, 1]):int(box[2, 1]), int(box[0, 0]):int(box[2, 0])]
                    height, width, channels = cropped_image.shape
                    num_char = round(width / height)
                    each_width = int(width / num_char)
                    start = 0
                    end = each_width
                    for char in range(num_char):
                        if char == num_char:
                            end = width
                        char_img = cropped_image[0 : height, start : end]
                        start = end
                        end += each_width
                        char_img = cv2.resize(char_img, (48,48))
                        ans = predict_katakana(char_img, model_recog)
                        results_text += key_list[int(np.argmax(ans))]

                    f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],results_text
                    ))
                    translated_txt = translator.translate(results_text, lang_tgt='en')
                    #cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                    img = cv2_putText_1(img = img, text = results_text, org = (box[0, 0], box[0, 1] - 20), fontFace = font, 
			                    color = (0, 0, 255), shape = [(box[0, 0], box[0, 1]), (box[2, 0], box[2, 1])])
                    cv2.putText(translated_img, translated_txt, ( box[0, 0], box[0, 1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(translated_img, (box[0, 0], box[0, 1]), (box[2, 0],box[2, 1 ]), (0, 255, 0), 2 )    
        
        img_path = os.path.join(FLAGS.output_dir, os.path.basename(img_file))
        img_path_translated = os.path.join(FLAGS.output_dir, "translated_" + os.path.basename(img_file))
        
        cv2.imwrite(img_path, img[:, :, ::-1])
        cv2.imwrite(img_path_translated, translated_img)

if __name__ == '__main__':
    main()
