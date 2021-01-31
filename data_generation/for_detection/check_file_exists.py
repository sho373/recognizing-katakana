import os
from shutil import copyfile

label_xml_dir = "synth_text_images2"
num_list = []

count = 1
for filename in os.listdir(label_xml_dir):
    if filename.endswith('.jpg'):
        #path = os.path.join(label_xml_dir, "gt_img_{}.txt".format(count))
        path = os.path.join(label_xml_dir, "img_{}.jpg".format(count))
        if not os.path.exists(path):
            print(count)
        count +=1