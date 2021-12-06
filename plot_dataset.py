import glob
import os.path
import time

import numpy as np

from functional.util.tools.args_util import parse_plot_opt
from functional.util.tools.draw_util import ImageDrawer
from functional.util.tools.file_util import create_dir, remove_dir
from tqdm import tqdm

global_label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

if __name__ == '__main__':
    # Get the configs from the command line
    opt = parse_plot_opt()
    # Generate the run time as the identifier
    run_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    try:
        image_path = glob.glob(os.path.join(opt.data_dir, "*data*.npy"))
        label_path = glob.glob(os.path.join(opt.data_dir, "*label*.npy"))
        if len(image_path) != len(label_path) or len(image_path) != 1:
            print("Multi-implementation of dataset!")
            raise ValueError
        data = np.load(image_path[0], allow_pickle=True)
        label = np.load(label_path[0], allow_pickle=True)
    except (IOError, TypeError, FileNotFoundError, ValueError, Exception) as e:
        print(e)
        raise e
    # Create the output directory, e.g. "output/base/2021123_223310"
    create_dir(opt.output_dir, opt.log_name, run_time)
    target_dir = os.path.join(opt.output_dir, opt.log_name, run_time)
    try:
        with open(os.path.join(target_dir, "source.txt"), "w", encoding="utf-8") as fout:
            fout.write(opt.data_dir)
        assert len(data) == len(label)
        assert data.shape[1:] == (32, 32, 3)
        assert label.shape[1:] == (10,)
        length = len(label)
        image_list = []
        label_list = []
        i = 0
        bar = tqdm(range(length))
        bar.set_description("Generating images")
        while i < length:
            image_list.append(data[i])
            label_list.append(label[i])
            if len(image_list) >= opt.batch_size:
                image_list = np.array(image_list)

                title = []
                for k in label_list:
                    classes = ""
                    for j in np.where(k > 0)[0]:
                        classes += "{}:{:.2f}\n".format(global_label[j], k[j])
                    title.append(classes)
                drawer = ImageDrawer(figsize=(6, 6))
                drawer.draw_same_batch(image_list, opt.row, title=title)
                drawer.save_image(os.path.join(target_dir, "{}.jpg".format(i)))
                image_list = []
                label_list = []
            i += 1
            bar.update(1)

        if len(image_list) > 0:
            image_list = np.array(image_list)
            title = []
            for i in label_list:
                classes = ""
                for j in np.where(i > 0)[0]:
                    classes += "{}:{:.2f}\n".format(global_label[j], i[j])
                title.append(classes)
            drawer = ImageDrawer()
            drawer.draw_same_batch(image_list, opt.row, title=title)
            drawer.save_image(os.path.join(target_dir, "{}.jpg".format(i)))
            drawer.clear()
    except (Exception, IOError, FileNotFoundError, KeyboardInterrupt) as e:
        remove_dir(opt.output_dir, opt.log_name, run_time)
        raise e
