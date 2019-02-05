# -*- encoding: utf-8 -*-
import os, base64, json

MODEL = "Reg"

MODEL_SAVE_PATH = "Reg_Model"
MODEL_NAME = "reg_model"
SUMMARY_PATH = "Reg_Logs"

MODEL_SAVE_PATH = MODEL + "/" + MODEL_SAVE_PATH
SUMMARY_PATH = SUMMARY_PATH + "/" + SUMMARY_PATH

# print(MODEL_SAVE_PATH)
# for name in os.listdir(MODEL_SAVE_PATH):
#     if ".data" in name:
#         print(name)

def transform_image_to_base46(category):
    #用来转换图片为base64编码，这样的编码可以用来放进modeldepot/percept里面训练。
    batch_images = []
    file = open("C:\\Users\\XMK23\\Pictures\\Weapon_Train_Images\\{}\\text.txt".format(category), "w")
    for name in os.listdir("C:\\Users\\XMK23\\Pictures\\Weapon_Train_Images\\{}".format(category)):
        f = open(os.path.join("C:\\Users\\XMK23\\Pictures\\Weapon_Train_Images\\{}".format(category),
                              name), "rb")
        ls_f = base64.b64encode(f.read())  # 读取文件内容，转换为base64编码
        f.close()
        _ = {}
        _["image"] = ls_f
        batch_images.append(_)
    json.dump(batch_images, file, indent=2)
    file.close()

transform_image_to_base46("miao dao")
