import os

MODEL = "Reg"

MODEL_SAVE_PATH = "Reg_Model"
MODEL_NAME = "reg_model"
SUMMARY_PATH = "Reg_Logs"

MODEL_SAVE_PATH = MODEL + "/" + MODEL_SAVE_PATH
SUMMARY_PATH = SUMMARY_PATH + "/" + SUMMARY_PATH

print(MODEL_SAVE_PATH)
for name in os.listdir(MODEL_SAVE_PATH):
    if ".data" in name:
        print(name)