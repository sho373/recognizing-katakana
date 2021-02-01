import os
import sys
import shutil
from detection import install_helper

print("Creating directories")

font_dir = "font"
img_dir = "check"
results_dir = "results"
model_dir = "model"

if not os.path.exists(img_dir):
    os.mkdir(img_dir)
if not os.path.exists(font_dir):
    os.makedirs(font_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

x = input("Do you want to install all necessary elements by pip?[y/n]")
if x == "y":
    os.system(f'{sys.executable} -m pip install -r requirements.txt')

###  Download dataset from google drive
x = input("Do you want to download trained model?[y/n]")
if x == "y":

    print("Downloading model")
    file_id = "1bzRC8xSLA_6qgEUD44hfFuhf8BUVED2L"
    katakana_model_data = "katakana_char_model.h5"
    install_helper.download_file_from_google_drive(file_id, katakana_model_data)
    
    file_id = "1wU9e3-UyRwrp_a5M5ALaqASDezavT1_j"
    model_data = "model/model-35.h5"
    install_helper.download_file_from_google_drive(file_id,  model_data)

    file_id = "1qM0Aoo3bz982ps3QxVaUBxOLADEv-1I-"
    model_json_data = "model/model.json"
    install_helper.download_file_from_google_drive(file_id,  model_json_data)

    print("Downloading japanese font")
    file_id = "1mbtGYV1VRoMSCmGbOMm11LmQ_CRL0vRm"
    font = "font/MSGOTHIC.TTC"
    install_helper.download_file_from_google_drive(file_id, font)

x = input("Do you want to download images for check?[y/n]")
if x == "y":
    
    print("Downloading check image")
    file_id = "1OyWnzGz9-gxjGGKVGK4yY5mfv9g7LZYv"
    check_rar_dataset = "check/images.rar"
    install_helper.download_file_from_google_drive(file_id, check_rar_dataset)
    
    install_helper.unrar(check_rar_dataset, img_dir)
    if os.path.exists(check_rar_dataset):
        os.remove(check_rar_dataset)