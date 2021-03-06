import glob
import os
import shutil
import time
import zipfile

from functional.util.tools.args_util import parse_pack_opt
from functional.util.tools.file_util import create_dir, remove_dir

if __name__ == '__main__':
    # Parse arguments
    opt = parse_pack_opt()
    # Create the identifier
    upload_identifier = time.strftime("%Y%m%d", time.localtime())
    # Create the base directory
    create_dir("upload")
    # Find the proper id for upload files
    files = glob.glob(os.path.join("upload", "{}*".format(upload_identifier)))
    current = len(files) + 1
    # Create target dataset
    create_dir("upload", "{}_{}".format(upload_identifier, current))
    target_path = os.path.join("upload", "{}_{}.".format(upload_identifier, current))

    try:
        # Get the data and the labels
        image_path = glob.glob(os.path.join(opt.data_dir, "*data*.npy"))
        label_path = glob.glob(os.path.join(opt.data_dir, "*label*.npy"))
        if len(image_path) != len(label_path) or len(image_path) != 1:
            print("Multi-implementation of dataset!")
            raise ValueError
        # Copy to target
        shutil.copy2(image_path[0], os.path.join(target_path, "data.npy"))
        shutil.copy2(label_path[0], os.path.join(target_path, "label.npy"))

    except (IOError, FileNotFoundError, FileExistsError, OSError, TypeError) as e:
        print("Unable to move data from {} to {}.".format(opt.data_dir, target_path))
        remove_dir("upload", "{}_{}".format(upload_identifier, current))
        raise e
        exit(-1)
    try:
        # Copy and rename the config to target
        shutil.copy2(os.path.join(opt.log_dir, opt.log_name, opt.identifier, "train_config.py"),
                     os.path.join(target_path, "config.py"))

    except (IOError, FileNotFoundError, FileExistsError, OSError, TypeError) as e:
        print("Unable to move config from {} to {}.".format(
            os.path.join(opt.log_dir, opt.log_name, opt.identifier, "train_config.py"),
            os.path.join(target_path, "config.py")))
        remove_dir("upload", "{}_{}".format(upload_identifier, current))
        raise e
        exit(-1)

    try:
        # Copy the weight files to target
        shutil.copy2(os.path.join(opt.output_checkpoint_dir, opt.log_name, opt.identifier, "preactresnet18.pth.tar"),
                     target_path)
        shutil.copy2(os.path.join(opt.output_checkpoint_dir, opt.log_name, opt.identifier, "wideresnet.pth.tar"),
                     target_path)

    except (IOError, FileNotFoundError, FileExistsError, OSError, TypeError) as e:
        print("Unable to move weight from {} to {}".format(
            os.path.join(opt.output_checkpoint_dir, opt.log_name, opt.identifier),
            target_path))
        remove_dir("upload", "{}_{}".format(upload_identifier, current))
        raise e
        exit(-1)

    # Generate the Dataset.zip
    print("Generating the Dataset.zip ......".format(target_path))
    zfile = zipfile.ZipFile(os.path.join(target_path, "Dataset.zip"), "w")
    # Change the working directory
    os.chdir(target_path)
    zfile.write("data.npy")
    zfile.write("label.npy")
    zfile.write("config.py")
    zfile.write("preactresnet18.pth.tar")
    zfile.write("wideresnet.pth.tar")
    zfile.close()
    # Finish
    print("Finish generate the Dataset.zip in {}.".format(target_path))
