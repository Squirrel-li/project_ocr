import os
import cv2
from pathlib import Path
#from paddleocr import PaddleOCR
#import paddlelite.lite


#import camera_ctrl
#import camera_detect
import test_opt

if __name__ == '__main__':
    IP_NUM = "10.244.18.107"
    cam_num = 0
    cnt = 0
    
    #define path
    abs_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(abs_path)
    path_project_root = os.path.join(dir_path, "..")
    path_project_root = os.path.normpath(path_project_root)
    
    path_dir_images = os.path.join(path_project_root, "images")

    path_dir_img_cut = os.path.join(path_dir_images, "carplates")
    path_dir_img_preprocess = os.path.join(path_dir_images, "preprocess_out")
    path_dir_text_ocr = os.path.join(path_dir_images, "OCR_out")
    
    print("begin")

    for i in range(8):
        cam_num = i