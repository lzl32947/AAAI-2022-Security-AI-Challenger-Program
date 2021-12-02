import glob
import os
import shutil
import time
import zipfile
from util.tools.args_util import parse_pack_opt
from util.tools.file_util import create_dir, remove_dir

if __name__ == '__main__':
    opt = parse_pack_opt()
    upload_identifier = time.strftime("%Y%m%d", time.localtime())
    files = glob.glob(os.path.join("upload", "{}*".format(upload_identifier)))
    current = len(files) + 1
    create_dir("upload", "{}_{}".format(upload_identifier, current))
    target_path = os.path.join("upload", "{}_{}.".format(upload_identifier, current))
    try:
        shutil.copy2(os.path.join(opt.data_dir, "data.npy"), target_path)
        shutil.copy2(os.path.join(opt.data_dir, "label.npy"), target_path)

    except IOError or FileNotFoundError or FileExistsError or OSError or TypeError:
        print("Unable to move data from {} to {}.".format(opt.data_dir, target_path))
        remove_dir("upload", "{}_{}".format(upload_identifier, current))
        exit(-1)
    try:
        shutil.copy2(os.path.join(opt.log_dir, opt.log_name, opt.identifier, "train_config.py"),
                     os.path.join(target_path, "config.py"))

    except IOError or FileNotFoundError or FileExistsError or OSError or TypeError:
        print("Unable to move config from {} to {}.".format(
            os.path.join(opt.log_dir, opt.log_name, opt.identifier, "train_config.py"),
            os.path.join(target_path, "config.py")))
        remove_dir("upload", "{}_{}".format(upload_identifier, current))
        exit(-1)

    try:
        shutil.copy2(os.path.join(opt.output_checkpoint_dir, opt.log_name, opt.identifier, "densenet121.pth.tar"),
                     target_path)
        shutil.copy2(os.path.join(opt.output_checkpoint_dir, opt.log_name, opt.identifier, "resnet50.pth.tar"),
                     target_path)

    except IOError or FileNotFoundError or FileExistsError or OSError or TypeError:
        print("Unable to move weight from {} to {}".format(
            os.path.join(opt.output_checkpoint_dir, opt.log_name, opt.identifier),
            target_path))
        remove_dir("upload", "{}_{}".format(upload_identifier, current))
        exit(-1)

    print("Generating the Dataset.zip ......".format(target_path))
    zfile = zipfile.ZipFile(os.path.join(target_path, "Dataset.zip"), "w")
    zfile.write(os.path.join(target_path, "data.npy"))
    zfile.write(os.path.join(target_path, "label.npy"))
    zfile.write(os.path.join(target_path, "config.py"))
    zfile.write(os.path.join(target_path, "densenet121.pth.tar"))
    zfile.write(os.path.join(target_path, "resnet50.pth.tar"))
    zfile.close()

    print("Finish generate the Dataset.zip in {}.".format(target_path))
