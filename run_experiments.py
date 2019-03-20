import sys, os, csv

def print_help():
    print('Usage:')
    print('./run_experiments.py <experiments_filename>')


def get_filename_from_args():
    num_args = len(sys.argv)

    if num_args != 2:
        print_help()
        sys.exit()

    filename = sys.argv[1]
    isfile = os.path.isfile(filename)

    if not isfile:
        print('File {} does not exist'.format(filename))
        sys.exit()

    return filename


def get_experiments(filename):
    experiments = []

    with open(filename, mode='r') as experiments_file:
        csv_dict_reader = csv.DictReader(experiments_file)

        for row in csv_dict_reader:
            experiments.append(row)

    print('Read {} experiments'.format(len(experiments)))

    return experiments

def run_experiment(exp_params):
    exp_num = exp_params['experiment_num']
    print('\n================= Running experiment no. {}  ================= \n'.format(exp_num))

    args = ""

    for arg_name, arg_val in exp_params.items():
        # print("arg_name={}, arg_val={}".format(arg_name, arg_val))
        args = args + " --" + arg_name + " " + arg_val

    command = "ipython LANL_NB.ipynb --" + args
    print(command)

    os.system(command)

    print('Finished running experiment no. {} ================= \n'.format(exp_num))


if __name__ == "__main__":
    filename = get_filename_from_args()

    experiments = get_experiments(filename)
    #print(experiments)

    print('\n================= Started running experiments ================= \n')

    for exp in experiments:
        run_experiment(exp)

    print('\n================= Finished running {} experiments ================= \n'.format(len(experiments)))


