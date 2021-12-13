import os
import sys


def file_out(case_output, data_input, fname, T, i, param, timeskip):

    if os.path.exists(data_input.data_directory):
        if not os.path.exists(data_input.dest_dir + '/' + case_output + '/' + data_input.case_dir):
            print(data_input.dest_dir + '/' + case_output + '/' + data_input.case_dir)
            os.makedirs(data_input.dest_dir + '/' + case_output + '/' + data_input.case_dir)
        case_path = data_input.dest_dir + '/' + case_output + '/' + data_input.case_dir
        fout = fname + '_' + "{0:.2f}".format(T[i] + timeskip) + '.tsv'
        if not os.path.exists(case_path + '/' + fout):
            f = open(case_path + '/' + fout, "a+")
            for ilist in range(len(param)):
                f.write("\t" + param[ilist] + "\t")
            f.write("\n")
            f.close()
    else:
        sys.exit("The case directory does not exist, please check the input file")
    return fout, case_path
