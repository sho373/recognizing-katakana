import os
import sys
import config
import shutil
import install_helper

x = input("Do you want to install all necessary elements by pip?[y/n]")
if x == "y":
    os.system(f'{sys.executable} -m pip install -r requirements.txt')

print("Creating directories")
data_dir = config.DATA_DIR
train_dir = config.TRAIN_IMG_DIR
test_dir = config.TEST_IMG_DIR
results_dir = config.RESULTS_DIR
check_dir = config.CHECK_DIR
model_dir = config.MODEL_DIR
char_dir = "../recognition/char_dir"

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
if not os.path.exists(check_dir):
    os.mkdir(check_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(char_dir):
    os.mkdir(char_dir)

###  Download dataset from google drive nad unpack it
x = input("Do you want to download images?[y/n]")
if x == "y":
    
    if not os.path.exists(config.TRAIN_IMG_DIR):
        os.mkdir(config.TRAIN_IMG_DIR)
        print("Downloading training image and labels")
        file_id = "1syT9Fi7LVRTYdDXDqLJCrCFlCS7r2rer"
        train_rar_dataset = "data/train/images.rar"
        install_helper.download_file_from_google_drive(file_id, train_rar_dataset)
    if not os.path.exists(config.TEST_IMG_DIR):
        os.mkdir(config.TEST_IMG_DIR)
        print("Downloading test image and labels")
        file_id = "1CH9CTx6FGB9A9lpeoN1xEqsGWaER0I3a"
        test_rar_dataset = "data/test/images.rar"
        install_helper.download_file_from_google_drive(file_id, test_rar_dataset)
    print("Downloading check image")
    file_id = "1eCAHQ8XjbqpzDz_6OKQRbaX5YIBtr359"
    check_rar_dataset = "check/images.rar"
    install_helper.download_file_from_google_drive(file_id, check_rar_dataset)

    print("Downloading images for recogntion")
    file_id = "1CY-oU0Em0GtNqLpJ7hK5sPwnclKc0XOb"
    char_rar_dataset = "../recognition/char_dir/images.rar"
    install_helper.download_file_from_google_drive(file_id, char_rar_dataset)

x1 = input("Do you want to unpack archive?[y/n]")
if x1 == "y":
    
    train_rar_dataset = "data/train/images.rar"
    install_helper.unrar(train_rar_dataset, config.TRAIN_IMG_DIR)
    if os.path.exists(train_rar_dataset):
        os.remove(train_rar_dataset)

    test_rar_dataset = "data/test/images.rar"
    install_helper.unrar(test_rar_dataset, config.TEST_IMG_DIR)
    if os.path.exists(test_rar_dataset):
        os.remove(test_rar_dataset)
    
    char_rar_dataset = "../recognition/char_dir/images.rar"
    install_helper.unrar(char_rar_dataset, "../recognition/char_dir")
    if os.path.exists(char_rar_dataset):
        os.remove(char_rar_dataset)

print("Installation complete")

