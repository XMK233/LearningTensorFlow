# -*- coding: utf-8 -*-
import json, re
import numpy as np
from jsondiff import diff
from deepdiff import DeepDiff

def my_obj_pairs_hook(lst):
    #没用了
    result={}
    count={}
    for key,val in lst:
        if key in count:
            count[key]=1+count[key]
        else:
            count[key]=1
        if key in result:
            if count[key] > 2:
                result[key].append(val)
            else:
                result[key]=[result[key], val]
        else:
            result[key]=val
    return result

def formalize_json(file):
    # model graph保存的文件不是正常的json格式，
    # 这个方法将.meta转为正常json文件。
    p1 = re.compile("[a-zA-Z_]+: \"")
    p2 = re.compile("[a-zA-Z_]+ {")
    p3 = re.compile("[a-zA-Z_]+: [\w+-]+")

    lines = open(file, "r").readlines()
    new_line = ""
    for i in range(len(lines) - 1):
        line = lines[i]
        nline = lines[i + 1]
        m1 = p1.findall(line)
        m2 = p2.findall(line)
        m3 = p3.findall(line)
        l = line
        if l[-1] == "\n":
            l = l[0:-1]
        #remove the last "\n"

        if len(m2) > 0:
            ori = m2[0].split()[0]
            new = "\"%s\":" %(ori)
            l = l.replace(ori, new)
            new_line += l + "\n"
            continue
        elif len(m1) > 0:
            ori = m1[0].split(":")[0]
            new = "\"%s\"" % (ori)
            new1 = l.strip().replace((ori + ": "), "").replace("\\", ">|<") ##这里要慎重
            if ori == "tensor_content":
                new1 = "\"________\""##################new1.replace("\\", "_")
            l = new + ": " + new1
            new_line += l + ("," if nline.strip() != "}" else "") + "\n"
            continue
        elif len(m3) > 0:
            ori = m3[0].split(": ")
            new = ["\"%s\"" %(i) for i in ori]
            l = ": ".join(new)
            new_line += l + ("," if nline.strip() != "}" else "") + "\n"
            continue
        else:
            new_line += l + ("," if nline.strip() != "}" else "") + "\n"
    new_line += lines[-1]
    return json.loads("{" + new_line + "}", object_pairs_hook=my_obj_pairs_hook)

def find_from_dict(dict, string):
    # string looks like: "root['PyTorch_CPU_MKL_Notebook']['awsmpConfig']['operatingSystems']['AMAZONLINUX']['aadistributionName']",
        tmp = dict
        p = re.compile("\'\w+\'")
        for i in p.findall(string):
            tmp = tmp[i[1:-1]]
        return tmp

def show_diff_result(last_versions, curr_versions, result_json):
    # 能够把一些diff结果显示出来
    # result_json是他俩的原始diff结果的json形式
    keys = result_json.keys()
    for key in keys:
        if key == "dictionary_item_added":
            rms = result_json[key]["py/set"]
            tmp_dict = {}
            for rm in rms:
                tt = find_from_dict(curr_versions, rm)
                tmp_dict[rm] = tt
            result_json[key] = tmp_dict
            pass
        elif key == "dictionary_item_removed":
            ads = result_json[key]["py/set"]
            tmp_dict = {}
            for ad in ads:
                tt = find_from_dict(last_versions, ad)
                tmp_dict[ad] = tt
            result_json[key] = tmp_dict
            pass
        else:
            pass
    return result_json

def diff_models(model1_dir, model2_dir):
    model1 = formalize_json(model1_dir)
    model2 = formalize_json(model2_dir)
    result = json.loads(DeepDiff(model1, model2, ignore_order=True).json)
    with open("diff_result.json", "w") as dr:
        json.dump(show_diff_result(model1, model2, result), dr, sort_keys=True, indent=4)

if __name__ == "main":
    model1_dir = "C:\\Users\\XMK23\\PycharmProjects\\LearningTensorFlow\\7_model_pipeline\\diff\\model.ckpt.json"
    model2_dir = "C:\\Users\\XMK23\\PycharmProjects\\LearningTensorFlow\\7_model_pipeline\\diff\\model.ckpt_retrained.json"
    diff_models(model1_dir, model2_dir)

'''c = open("tmp.json", "w")
json.dump(formalize_json("model.ckpt.json"), c, indent=4)
c.close()'''



