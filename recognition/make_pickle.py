import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle


out_dir = "./char_dir" 
save_file = "./katakana.pickle"

im_size = 48
plt.figure(figsize=(9, 17))
katakaba_dir = list(range(0, 71)) #ーア～ン
result = []

for i, code in enumerate(katakaba_dir):
    img_dir = out_dir + "/" + str(code)
    fs = glob.glob(img_dir + "/*")
    print("dir=",  img_dir)

    for j, f in enumerate(fs):
        img = cv2.imread(f)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img_gray, (im_size, im_size))
        result.append([i, img])

        #show data
        if j == 3:
            plt.subplot(11, 8, i + 1)
            plt.axis("off")
            plt.title(str(i))
            plt.imshow(img, cmap='gray')


pickle.dump(result, open(save_file, "wb"))
plt.show()
