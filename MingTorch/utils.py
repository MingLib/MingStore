import os, sys
sys.path.append(os.getcwd())
from collections import namedtuple

model_path_set = namedtuple("PATHs","save_path record_path")

def set_log_file(fname):
    import subprocess
    # set log file
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def path_manager(arch_name, dataset_name, records_root="Records", models_root="Models"):
    py_root = os.getcwd()
    models_path = os.path.join(py_root, models_root, dataset_name)
    records_path = os.path.join(py_root, records_root, dataset_name)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(records_path, exist_ok=True)
    models_file_name = arch_name+".pth"
    records_file_name = arch_name + ".txt"
    models_file = os.path.join(models_path, models_file_name)
    recodes_file = os.path.join(records_path, records_file_name)
    return model_path_set(models_file, recodes_file)