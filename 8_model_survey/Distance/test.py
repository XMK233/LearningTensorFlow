import tensorflow as tf
from numpy import *
import re, json, os, scipy
from deepdiff import DeepDiff
import matplotlib.pyplot as plt

MODEL_SAVE_PATH = "Reg_Model"
MODEL_NAME = "model.ckpt"
SUMMARY_PATH = "Reg_Logs"

CNN_SAVE_PATH = "CNN_model/"
CNN_NAME = "cnn_model"
CNN_LOGS = "CNN_Logs"
KEEP_RATE = 0.8


def my_obj_pairs_hook(lst):
    result = {}
    count = {}
    for key, val in lst:
        if key in count:
            count[key] = 1 + count[key]
        else:
            count[key] = 1
        if key in result:
            if count[key] > 2:
                result[key].append(val)
            else:
                result[key] = [result[key], val]
        else:
            result[key] = val
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
        # remove the last "\n"

        if len(m2) > 0:
            ori = m2[0].split()[0]
            new = "\"%s\":" % (ori)
            l = l.replace(ori, new)
            new_line += l + "\n"
            continue
        elif len(m1) > 0:
            ori = m1[0].split(":")[0]
            new = "\"%s\"" % (ori)
            new1 = l.strip().replace((ori + ": "), "").replace("\\", ">|<")  ##这里要慎重
            if ori == "tensor_content":
                new1 = "\"________\""  ##################new1.replace("\\", "_")
            l = new + ": " + new1
            new_line += l + ("," if nline.strip() != "}" else "") + "\n"
            continue
        elif len(m3) > 0:
            ori = m3[0].split(": ")
            new = ["\"%s\"" % (i) for i in ori]
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
    result = json.loads(DeepDiff(model1, model2).json)  # , ignore_order=True

    # with open("diff_result.json", "w") as dr:
    #     json.dump(show_diff_result(model1, model2, result), dr, sort_keys=True, indent=4)

    return result


def variable_to_json(model_path):
    reader = tf.train.NewCheckpointReader(model_path)
    global_variables = reader.get_variable_to_shape_map()

    item = {}
    for variable_name in global_variables:
        item[variable_name] = {"shape": global_variables[variable_name],
                               "value": reader.get_tensor(variable_name).tolist()}

    # with open(NAME + ".json", "w") as v:
    #     json.dump(item, v, sort_keys= True, indent= 4)#

    return item

def diff_varialbes(model1_dir, model2_dir):
    return json.loads(DeepDiff(
        variable_to_json(model1_dir),
        variable_to_json(model2_dir)
    ).json)

def distance(v1, v2):
    return linalg.norm(array(v1) - array(v2))

def distances_changing(MODEL_SAVE_PATH, type):
    graph_dirs = []
    for name in os.listdir(MODEL_SAVE_PATH):
        if ".data" in name:
            graph_dirs.append(name)
    graph_dirs.sort()

    variables = {}
    for i in range(len(graph_dirs)):
        if type == "reg":
            m1 = ".".join(graph_dirs[i].split(".")[0:2])
        elif type == "cnn":
            m1 = graph_dirs[i].split(".")[0]
        else:
            m1 = "???"
        vars = variable_to_json(os.path.join(MODEL_SAVE_PATH, m1))
        keys = list(vars.keys())
        for key in keys:
            if key in variables.keys():
                variables[key].append(vars[key]["value"])
            else:
                variables[key] = [vars[key]["value"]]
    #print(variables)

    variables_diff = {}
    for key in list(variables.keys()):
        print(key)
        vecs = variables[key]
        for v in range(len(vecs) - 1):
            dis_mat = distance(vecs[v], vecs[v + 1])
            if key in variables_diff.keys():
                variables_diff[key].append(dis_mat)
                pass
            else:
                variables_diff[key] = [dis_mat]
                ####

    for key in variables_diff:
        values = variables_diff[key]
        print(key, values)
        plt.figure()
        plt.plot(range(len(values)), values)
        #plt.show()
        plt.savefig("%s.jpg" %(key.replace("/", "-")))

#distances_changing_reg(MODEL_SAVE_PATH, MODEL_NAME, "reg")
distances_changing(CNN_SAVE_PATH, "cnn")





# a = array([[[1,1,1], [2,2,2], [3,3,3]], [[4,4,4], [5,5,5], [6,6,6]], [[7,7,7], [8,8,8], [9,9,9]]])
# print(a)
# b = array([[[2,1,2],[3,2,3],[4,5,6]], [[2,3,5],[6,7,8],[7,7,6]], [[8,9,1],[9,3,4],[10,11,2]]])
# print(b)
# print(linalg.norm(a - b))


