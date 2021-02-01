import re
import numpy as np #!
from numpy import savetxt
import random
import cv2
import numpy as np
import matplotlib as plt
from PIL import ImageFont, ImageDraw, Image
import os
from rotate_text import draw_rotated_text
import math

img_w = 512
img_h = 512
#mp.dps = 0

def find_point(x, y) :
    x1 = 0
    y1 = 0
    x2 = img_w
    y2 = img_h

    if (x > x1 and x < x2 and 
        y > y1 and y < y2) : 
        return True
    else : 
        return False

def is_cor_inside_image(tl, tr, bl, br):
    if find_point(tl[0], tl[1]) and find_point(tr[0], tr[1]):
        if find_point(bl[0], bl[1]) and find_point(br[0],br[1]): 
            return True
    return False

# def calRotatedPoints(_x1,_y1,cx,cy,angle):
#     """
#     Function calculate the cordinate (x1,y1) for rotaed points of box
#     """
#     # x′=xcosθ-ysinθ
#     # y′=xsinθ+ycosθ
#     tempX = fsub(_x1,cx)
#     tempY = fsub(_y1,cy)
#     rotatedX = fsub(fmul(tempX,cos(angle)), fmul(tempY,sin(angle)))
#     rotatedY = fadd(fmul(tempX,sin(angle)), fmul(tempY,cos(angle)))
#     x1 = fadd(rotatedX,cx)
#     y1 = fadd(rotatedY,cy)
#     return int(x1),int(y1)

def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR

def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL

def cv2_putText(img, text, org, fontFace, color, shape):
	b, g, r = color
	x, y = org
	colorRGB = (r, g, b)
	imgPIL = cv2pil(img)
	draw = ImageDraw.Draw(imgPIL)
	#fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
	#draw.rectangle(shape, outline = (0,255,0), width = 2)
	draw.text(xy = (x,y), text = text, fill = colorRGB, font = fontFace)
	imgCV = pil2cv(imgPIL)
	return imgCV

def gen_random_par(words_list):
    words_index = random.randint(0,len(words_list) -1)
    font_scale = random.randint(30,80)
    horizontal = random.randint(0,2)
    words = words_list[words_index]
    words_len = len(words) - 1

    if words_len < 4:
        x_cor = random.randint(10,int(img_w/2))
    else :
        x_cor = random.randint(10,int(img_w/4))


    y_cor = random.randint(font_scale, int(img_h - font_scale))


    font = ImageFont.truetype("./fonts/MSGOTHIC.TTC", font_scale)

    tl = (x_cor, y_cor)
    tr = (int(x_cor + words_len*font_scale + font_scale), y_cor)
    br = (int(x_cor + words_len*font_scale + font_scale),int(y_cor + font_scale),)
    bl = (x_cor, int(y_cor + font_scale))

    return words, font, horizontal, font_scale, tl, tr, bl, br

def boxes_intersection(bb1, bb2):
    """
    check two bounding boxes intersect.

    """
    assert bb1[0][0] < bb1[1][0]
    assert bb1[0][1] < bb1[1][1]
    assert bb2[0][0] < bb2[1][0]
    assert bb2[0][1] < bb2[1][1]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0][0], bb2[0][0])
    y_top = max(bb1[0][1], bb2[0][1])
    x_right = min(bb1[1][0], bb2[1][0])
    y_bottom = min(bb1[1][1], bb2[1][1])

    if x_right < x_left or y_bottom < y_top:
        return False
    return True


    
def main():
    results_dir = "synth_text_images"
    #img_dir = "images"
    # results_dir = "results"
    img_dir = "images"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    words_list = np.zeros(shape=(1, 1),dtype=object)

    with open('katakana_words.txt', 'r', encoding='utf-8') as f:
        for line in f:
            words_list = np.append(words_list, line.replace("\n",""))
    words_list = np.delete(words_list,0)
    
    text_color_list = [(0, 0, 255),(0,0,0),(15, 8, 44), (0,0,0), (255,250,250), (255,0,0), (255,250,250), (255,127,0)] 
    #blue, black, dark blue, black, red, white,orange
    count = 0

    for filename in os.listdir(img_dir):
        if filename.endswith(".jpg"):
            
            image = cv2.imread(os.path.join(img_dir,filename))
            count+=1
            words_num = random.randint(1,3)
           
            cor_list = []

            if not image is None:
                image = cv2.resize(image, (img_w,img_h))
                
                for i in range(words_num):
                    is_not_cor_inside = True
                    is_not_box_intersect = True

                    if i == 0:
                        while is_not_cor_inside:
                            
                            words, font, horizontal, font_scale, tl, tr, bl, br = gen_random_par(words_list)
                            if is_cor_inside_image(tl,tr,bl,br):
                                is_not_cor_inside = False
                        cor_list.append([tl, br])   
                    elif i == 1:
                        while is_not_box_intersect:
                            while is_not_cor_inside:
                                words, font, horizontal, font_scale, tl, tr, bl, br = gen_random_par(words_list)
                                if is_cor_inside_image(tl,tr,bl,br):
                                    is_not_cor_inside = False
                            cor_list.append([tl, br])
                            check = boxes_intersection(cor_list[i-1], cor_list[i])
                            if check == False:
                                is_not_box_intersect = False
                            else:
                                cor_list.pop(-1)
                                is_not_cor_inside = True
                    elif i == 2:
                        while is_not_box_intersect:
                            while is_not_cor_inside:
                                words, font, horizontal, font_scale, tl, tr, bl, br = gen_random_par(words_list)
                                if is_cor_inside_image(tl,tr,bl,br):
                                    is_not_cor_inside = False
                            cor_list.append([tl, br])        
                            check_2 = boxes_intersection(cor_list[0], cor_list[i])
                            check_3 = boxes_intersection(cor_list[1], cor_list[i])
                            if check_2 == False and check_3 == False:
                                is_not_box_intersect = False      
                            else:
                                cor_list.pop(-1)
                                is_not_cor_inside = True

                    if not horizontal:
                        image = Image.fromarray(image)
                        
                        angle = random.randint(-15,15)
                        draw_rotated_text(image, angle, (abs(int(tl[0])), abs(int(tl[1] - font_scale*0))), words, 
                                            random.choice(text_color_list), font=font)

                        image = np.array(image)
                        #cv2.rectangle(image, tl, br, (0, 255, 0), 2)
                        # cv2.imshow("text", image)
                        # cv2.waitKey(0)
                    else :
                        image = cv2_putText(img = image, text = words, 
                                    org = (tl[0], tl[1] - font_scale*0.15 ), fontFace = font, 
                                    color = random.choice(text_color_list), shape = [tl, br])
                        #cv2.rectangle(img, tl, br, (0, 255, 0), 2)            
                        # cv2.imshow("text", image)
                        # cv2.waitKey(0)

                name = os.path.join(results_dir,"img_{}.jpg".format(count))
                cv2.imwrite(name, image)
                
                print(name)

if __name__ == "__main__":
    main()