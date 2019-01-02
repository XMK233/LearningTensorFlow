import json, re
import numpy as np
from jsondiff import diff
from deepdiff import DeepDiff

def formalize_json(file):
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
            new1 = l.strip().replace((ori + ": "), "")
            #new1 = l.split(": ")[1]
            if ori == "tensor_content":
                q = np.fromstring(new1[1:-1], dtype=np.int32)
                new1 = "\"" + (",".join(str(_) for _ in q)) + "\""
                #print(new1)
            l = new + ": " + new1 #l.replace(ori, new)
            new_line += l + ("," if nline.strip() != "}" else "") + "\n"
            continue
        elif len(m3) > 0:
            ori = m3[0].split(": ")
            new = ["\"%s\"" %(i) for i in ori]
            l = ": ".join(new)
            '''for _ in range(len(new)):
                l = l.replace(ori[_], new[_])'''
            new_line += l + ("," if nline.strip() != "}" else "") + "\n"
            continue
        else:
            new_line += l + ("," if nline.strip() != "}" else "") + "\n"
    new_line += lines[-1]
    with open("tmp.json", "w") as c:
        c.write("{%s}"%(new_line))
    return json.loads("{%s}"%(new_line))

print(formalize_json("mnist_model-2018-12-31-22-35-28.json"))



'''file = ["mnist_model-2018-12-31-22-35-28.json", "mnist_model-2018-12-31-22-37-04.json"]
result1 = DeepDiff(formalize_json(file[0]), formalize_json(file[1]))
print(result1)'''

'''
text = open("txt.json", "r").read()
for i in p2.findall(text):
    ori = i.split()[0]
    new = "\"%s\":" %(ori)
    text = text.replace(ori, new)
for i in p1.findall(text):
    ori = i.split(":")[0]
    new = "\"%s\"" %(ori)
    text = text.replace(ori, new)
print(text)'''
'''text = []
file = ["mnist_model-2018-12-31-22-35-28.json", "mnist_model-2018-12-31-22-37-04.json"]
for i in range(2):
    f = open(file[i], "r")
    text = f.read()
    f.close()

    p1 = re.compile("\w+: \"")
    p2 = re.compile("\w+ {")'''

'''last_version = json.load()
current_version = json.load()
result1 = DeepDiff(last_version, current_version)
result2 = diff(last_version, current_version)

print(result1.json)
print(result2)'''


