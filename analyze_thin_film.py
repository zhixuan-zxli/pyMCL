import numpy as np
from fem.post import printConvergenceTable

def getTimeConvergence() -> None:
    filenames = "result/tf-1e-2-uni-256-T{}/0032.npz"
    data = []
    table_headers = ["T{}".format(i) for i in range(3)]
    error_table = { "h inf": [], "g inf": [], "a": [] }
    for i in range(4):
        data.append(np.load(filenames.format(i)))
    for i in range(3):
        h_diff = data[i+1]["h"] - data[i]["h"]
        error_table["h inf"].append(np.linalg.norm(h_diff[2:-1], ord=np.inf))
        g_diff = data[i+1]["g"] - data[i]["g"]
        error_table["g inf"].append(np.linalg.norm(g_diff[2:-1], ord=np.inf))
        a_diff = data[i+1]["a"] - data[i]["a"]
        error_table["a"].append(np.abs(a_diff).item())
    printConvergenceTable(table_headers, error_table)

if __name__ == "__main__":
    getTimeConvergence()