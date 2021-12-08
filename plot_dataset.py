import glob
import os.path
import time

import numpy as np

from functional.util.tools.args_util import parse_plot_opt
from functional.util.tools.draw_util import ImageDrawer, label_dict2str
from functional.util.tools.file_util import create_dir, remove_dir
from tqdm import tqdm

from functional.util.tools.trainer_util import get_label_name, get_class

global_label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def plot_same_class(args, data_array, label_array, save_path):
    length = len(label_array)

    classes_index_label = [(i, j, k) for i, j, k in zip(get_class(label_array), label_array, data_array)]
    classes_index_label.sort(key=lambda x: x[0])

    data_array = np.array([i[2] for i in classes_index_label])
    label_array = np.array([i[1] for i in classes_index_label])
    classes = np.array([i[0] for i in classes_index_label])

    # Enable tqdm for showing
    bar = tqdm(range(length))
    bar.set_description("Generating images")
    for c in range(0, len(global_label)):
        image_list = []
        label_list = []
        data_sub = data_array[classes == c]
        label_sub = label_array[classes == c]
        i = 0
        while i < len(data_sub):
            image_list.append(data_array[i])
            label_list.append(label_array[i])
            # plot for batch, one image contains "batch_size" numbers of images
            if len(image_list) >= args.batch_size:
                image_list = np.array(image_list)

                title = label_dict2str(get_label_name(np.array(label_list)))

                # Create Drawer
                drawer = ImageDrawer(figsize=(6, 6))
                drawer.draw_same_batch(image_list, args.row, title=title)
                drawer.save_image(os.path.join(save_path, "{}-{}.jpg".format(global_label[c], i)))
                drawer.clear()
                del drawer
                # Finish drawing
                image_list = []
                label_list = []
            i += 1
            bar.update(1)

        # Handle the rest images
        if len(image_list) > 0:
            image_list = np.array(image_list)
            title = label_dict2str(get_label_name(np.array(label_list)))
            # Create drawer
            drawer = ImageDrawer()
            drawer.draw_same_batch(image_list, args.row, title=title)
            drawer.save_image(os.path.join(save_path, "{}-{}.jpg".format(global_label[c], i)))
            drawer.clear()
            del drawer
    # Finish
    bar.close()


def plot_sequentially(args, data_array, label_array, save_path):
    length = len(label_array)
    image_list = []
    label_list = []
    i = 0
    # Enable tqdm for showing
    bar = tqdm(range(length))
    bar.set_description("Generating images")
    while i < length:
        image_list.append(data_array[i])
        label_list.append(label_array[i])
        # plot for batch, one image contains "batch_size" numbers of images
        if len(image_list) >= args.batch_size:
            image_list = np.array(image_list)

            title = label_dict2str(get_label_name(np.array(label_list)))

            # Create Drawer
            drawer = ImageDrawer(figsize=(6, 6))
            drawer.draw_same_batch(image_list, args.row, title=title)
            drawer.save_image(os.path.join(save_path, "{}.jpg".format(i)))
            drawer.clear()
            del drawer
            # Finish drawing
            image_list = []
            label_list = []
        i += 1
        bar.update(1)

    # Handle the rest images
    if len(image_list) > 0:
        image_list = np.array(image_list)
        title = label_dict2str(get_label_name(np.array(label_list)))
        # Create drawer
        drawer = ImageDrawer()
        drawer.draw_same_batch(image_list, args.row, title=title)
        drawer.save_image(os.path.join(save_path, "{}.jpg".format(i)))
        drawer.clear()
        del drawer
    # Finish
    bar.close()


if __name__ == '__main__':
    # Get the configs from the command line
    opt = parse_plot_opt()
    # Generate the run time as the identifier
    run_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    try:
        # Get the data and the label
        image_path = glob.glob(os.path.join(opt.data_dir, "*data*.npy"))
        label_path = glob.glob(os.path.join(opt.data_dir, "*label*.npy"))
        if len(image_path) != len(label_path) or len(image_path) != 1:
            print("Multi-implementation of dataset!")
            raise ValueError
        data_ = np.load(image_path[0], allow_pickle=True)
        label_ = np.load(label_path[0], allow_pickle=True)
    except (IOError, TypeError, FileNotFoundError, ValueError, Exception) as e:
        print(e)
        raise e
    # Create the output directory, e.g. "output/base/2021123_223310"
    create_dir(opt.output_dir, opt.log_name, run_time)
    target_dir = os.path.join(opt.output_dir, opt.log_name, run_time)
    try:
        # Write the source of the dataset
        with open(os.path.join(target_dir, "source.txt"), "w", encoding="utf-8") as fout:
            fout.write(opt.data_dir)
        assert len(data_) == len(label_)
        assert data_.shape[1:] == (32, 32, 3)
        assert label_.shape[1:] == (10,)
        if opt.class_first:
            plot_same_class(opt, data_, label_, target_dir)
        else:
            plot_sequentially(opt, data_, label_, target_dir)

    except (Exception, IOError, FileNotFoundError, KeyboardInterrupt) as e:
        remove_dir(opt.output_dir, opt.log_name, run_time)
        raise e
