#
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=128,height=128):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("/home/seasun/LearningTF/6_Model_Versoin/Weapon_Train_Images/huan shou dao/*.jpg"):
    convertjpg(jpgfile,"/home/seasun/LearningTF/6_Model_Versoin/train_images/huanshoudao")
for jpgfile in glob.glob("/home/seasun/LearningTF/6_Model_Versoin/Weapon_Train_Images/han jian/*.jpg"):
    convertjpg(jpgfile,"/home/seasun/LearningTF/6_Model_Versoin/train_images/hanjian")
for jpgfile in glob.glob("/home/seasun/LearningTF/6_Model_Versoin/Weapon_Train_Images/miao dao/*.jpg"):
    convertjpg(jpgfile,"/home/seasun/LearningTF/6_Model_Versoin/train_images/miaodao")
for jpgfile in glob.glob("/home/seasun/LearningTF/6_Model_Versoin/Weapon_Train_Images/qing jian/*.jpg"):
    convertjpg(jpgfile,"/home/seasun/LearningTF/6_Model_Versoin/train_images/qingjian")
for jpgfile in glob.glob("/home/seasun/LearningTF/6_Model_Versoin/Weapon_Train_Images/yan ling dao/*.jpg"):
    convertjpg(jpgfile,"/home/seasun/LearningTF/6_Model_Versoin/train_images/yanlingdao")