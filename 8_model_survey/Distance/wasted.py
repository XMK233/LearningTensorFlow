def distances_changing_reg(MODEL_SAVE_PATH, MODEL_NAME, type = "reg"):
    def distance(v1, v2):
        # 我希望这个variables是列表形式的。
        # Euclidean Distance
        vector1 = mat(v1)
        vector2 = mat(v2)
        return sqrt((vector1 - vector2) * (vector1 - vector2).T)

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

    print(variables)


    variables_diff = {}
    for key in list(variables.keys()):
        vecs = variables[key]
        for v in range(len(vecs) - 1):
            dis_mat = distance(vecs[v], vecs[v + 1]).tolist()
            if key in variables_diff.keys():
                variables_diff[key].append(dis_mat[0][0])
                pass
            else:
                variables_diff[key] = [dis_mat[0][0]]
                ####

    for key in variables_diff:
        values = variables_diff[key]
        print(key, values)
        plt.figure()
        plt.plot(range(len(values)), values)
        #plt.show()
        plt.savefig("%s.jpg" %(key))