import os
import splitfolders
import shutil

input_dir =  "synth_text_images"
out_dir = "data"
txt_dir = "txt"

if not os.path.exists(out_dir):
        os.mkdir(out_dir)

splitfolders.ratio(input_dir, output=out_dir,ratio=(.8, 0.2),group_prefix=None)

train_img_names = os.listdir(os.path.join(out_dir, "train/images"))
val_img_names = os.listdir(os.path.join(out_dir, "val/images"))

for file_name in train_img_names:
    txt_name = 'gt_' + file_name.split('.')[0] + '.txt'
    shutil.move(os.path.join(txt_dir, txt_name),os.path.join(out_dir, "train"))
    shutil.move(os.path.join("data/train/images", file_name), os.path.join(out_dir, "train"))
   

shutil.rmtree("data/train/images")

for file_name in val_img_names:
    txt_name = 'gt_' + file_name.split('.')[0] + '.txt'
    shutil.move(os.path.join(txt_dir, txt_name),os.path.join(out_dir, "val"))
    shutil.move(os.path.join("data/val/images", file_name), os.path.join(out_dir, "val"))

shutil.rmtree("data/val/images")
