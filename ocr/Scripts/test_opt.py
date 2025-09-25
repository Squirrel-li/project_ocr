import os
import cv2
from pathlib import Path
from paddlelite.lite import *

def test_opt(path_input, path_output):
	
	opt = Opt()# 1. 创建opt实例
	opt.set_model_dir(path_input)# 2. 指定输入模型地址 "./mobilenet_v1"
	opt.set_valid_places("arm")# 3. 指定转化类型： arm、x86、opencl、npu
	opt.set_model_type("naive_buffer")# 4. 指定模型转化类型： naive_buffer、protobuf
	opt.run()# 5. 执行模型优化
	opt.set_optimize_out(path_output)# 4. 输出模型地址"mobilenetv1_opt"

if __name__ == "__main__":

    abs_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(abs_path)
    path_project_root = os.path.join(dir_path, "..")
    path_project_root = os.path.normpath(path_project_root)
    print(path_project_root)

    path_input_modle = os.path.join(path_project_root, "en_number_mobile_v2.0_rec_slim_infer")
    print(path_input_modle)
    path_output_modle = os.path.join(path_project_root, "Lite_model")
    print(path_output_modle)
    test_opt(path_input_modle, path_output_modle)