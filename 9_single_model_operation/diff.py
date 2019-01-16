import tensorflow as tf
from numpy import *
import re, json, os, scipy
from deepdiff import DeepDiff
import matplotlib.pyplot as plt

MODEL = "Reg"

MODEL_SAVE_PATH = MODEL + "_Model"
MODEL_NAME = "reg_model"

MODEL_SAVE_PATH = MODEL + "/" + MODEL_SAVE_PATH


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
            new1 = l.strip().replace((ori + ": "), "")[1:-1]
            new1 = "\"%s\"" %(new1.replace("\\", ">|<").replace("\"", ">:<"))  ##这里要慎重
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
    result = json.loads("{" + new_line + "}", object_pairs_hook=my_obj_pairs_hook)
    return result


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

def distances_changing(MODEL_SAVE_PATH, MODEL):
    graph_dirs = []
    for name in os.listdir(MODEL_SAVE_PATH):
        if ".data" in name:
            graph_dirs.append(name)
    graph_dirs.sort()

    variables = {}
    for i in range(len(graph_dirs)):
        m1 = graph_dirs[i].split(".")[0]
        vars = variable_to_json(os.path.join(MODEL_SAVE_PATH, m1))
        keys = list(vars.keys())
        for key in keys:
            if key in variables.keys():
                variables[key].append(vars[key]["value"])
            else:
                variables[key] = [vars[key]["value"]]

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
        if not os.path.exists("Var_Diff/%s/" %(MODEL)):
            os.makedirs("Var_Diff/%s/" %(MODEL))
        plt.savefig("Var_Diff/%s/%s.jpg" %(MODEL, key.replace("/", "-")))

def graph_variable_diff(MODEL_SAVE_PATH, MODEL):
    graph_dirs = []
    for name in os.listdir(MODEL_SAVE_PATH):
        if ".json" in name:
            graph_dirs.append(name)

    if not os.path.exists("Model_Diff/%s/" %(MODEL)):
        os.makedirs("Model_Diff/%s/" %(MODEL))

    for i in range(len(graph_dirs) - 1):
        print("graph diff between ckpt %d and %d" % (i, i + 1))
        graph_diff = diff_models(os.path.join(MODEL_SAVE_PATH, graph_dirs[i]),
                                 os.path.join(MODEL_SAVE_PATH, graph_dirs[i + 1]))
        with open("Model_Diff/%s/graph_diff %d and %d.json" % (MODEL, i, i + 1), "w") as d:
            json.dump(graph_diff, d, sort_keys=True, indent=2)

        m1 = graph_dirs[i].split(".")[0]
        m2 = graph_dirs[i + 1].split(".")[0]
        print("variable diff between ckpt %d and %d" % (i, i + 1))
        variable_diff = diff_varialbes(os.path.join(MODEL_SAVE_PATH, m1),
                                       os.path.join(MODEL_SAVE_PATH, m2))
        with open("Model_Diff/%s/variable_diff %d and %d.json" % (MODEL, i, i + 1), "w") as d:
            json.dump(variable_diff, d, sort_keys=True, indent=2)

#distances_changing(MODEL_SAVE_PATH, MODEL)
graph_variable_diff(MODEL_SAVE_PATH, MODEL)


