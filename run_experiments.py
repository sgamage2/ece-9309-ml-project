import sys, os, csv
import subprocess

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
        csv_dict_reader = csv.DictReader(filter(lambda row: row[0]!='#', experiments_file))

        for row in csv_dict_reader:
            experiments.append(row)

    print('Read {} experiments'.format(len(experiments)))

    return experiments

def run_experiment(exp_params):
    exp_num = exp_params['experiment_num']
    print('\n================= Running experiment no. {}  ================= \n'.format(exp_num))

    # command_args = ""
    command_args = ["ipython", "LANL_NB.ipynb", "--"]
    results_dir = ""

    for arg_name, arg_val in exp_params.items():
        command_args.append("--" + arg_name)
        command_args.append(arg_val)

        if arg_name == 'results_dir':
            results_dir = arg_val

    assert len(results_dir) > 0

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_log_filename = results_dir + '/' + 'output.log'

    with open(output_log_filename, 'w') as output_log_file:
        p = subprocess.Popen(command_args, stdout=subprocess.PIPE)
        for line in iter(p.stdout.readline, b''):
            line_str = str(line.strip().decode('UTF-8'))
            print(line_str)
            output_log_file.write(line_str + '\n')
        p.stdout.close()
        p.wait()

    print('\n================= Finished running experiment no. {} ================= \n'.format(exp_num))


if __name__ == "__main__":
    filename = get_filename_from_args()

    experiments = get_experiments(filename)
    #print(experiments)

    print('\n================= Started running experiments ================= \n')

    for exp in experiments:
        run_experiment(exp)

    print('\n================= Finished running {} experiments ================= \n'.format(len(experiments)))


