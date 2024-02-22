import numpy as np

def printConvergenceTable(mesh_table, error_table):
    """ 
    Print the convergence table. 
    mesh_table: a list of string for mesh sizes as table headers. 
    error_table: a dict, "norm_type":[errors on each level].
    """
    m = len(mesh_table)
    # print the header
    header_str = "\n{0: <20}".format("")
    for i in range(m-1):
        header_str += "{0: <10}{1: <8}".format(mesh_table[i], "rate")
    header_str += "{0: <10}".format(mesh_table[-1])
    print(header_str)
    # print each norm
    for (norm_type, error_list) in error_table.items():
        error_str = "{0: <20}".format(norm_type)
        for i in range(m-1):
            error_str += "{0:<10.2e}{1:<8.2f}".format(error_list[i], np.log2(error_list[i]/error_list[i+1]))
        error_str += "{0:<10.2e}".format(error_list[-1])
        print(error_str)
    