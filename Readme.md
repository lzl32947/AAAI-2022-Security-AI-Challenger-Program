# Ali AAAI-2022 Security AI Challenger Program

This is the repo for Ali AAAI-2022 security AI Challenger program phase VIII(AAAI2022 安全AI挑战者计划第八期).

## Base

The files in ```base``` directory is the original directory for baseline files, and can be found
in [training_template_for_AI_challenger_sea8](https://github.com/vtddggg/training_template_for_AI_challenger_sea8)

These files should not be modified to overwrite the logic and running procedures.

Original ```Readme``` can be found in [here](doc/README.md).

## Run baseline

The main script of the project is ```main.py```, and the parsed args can be found at ```functional/util/tools/args_util.py```(
click [here](functional/util/tools/args_util.py))

Typically, can run the following commands:

```shell
python trainer.py --log_name base --data_train dataset/baseline --data_eval dataset/baseline --output_checkpoint checkpoint
```

Specifically, if you want to enable tensorboard for recording the images for training and testing, and watch the training procedures, please add the ```--enable_tensorboard``` in command line.

Commonly when modified the files, you can run the command like the followings:

```shell
python trainer.py --log_name cifar10 --data_train dataset/cifar_10_standard_train --data_eval dataset/cifar_10_standard_test --eval_per_epoch 20 --enable_tensorboard
```
## Label

The output label should be in one-hot of size of 10, which separately refer to:

```python
label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
```

## Log

Log files can be found in ```log``` directory.

## Pack for uploads

When training finished, all the files can be packed with the ```pack_upload.py```, which args can be found same at ```functional/util/tools/args_util.py```(
click [here](functional/util/tools/args_util.py))

Typically, can run the following commands:

```shell
python pack_upload.py --log_name cifar10 --identifier 20211128_213656 --data_dir dataset/baseline
```

Then you can find the packed zip file in ```upload``` directory.

## Generate dataset

When generating the dataset:

1. You should select a base dataset, which should be defined as a function in ```functional/generator_function/dataset_function.py```(click [here](functional/generator_function/dataset_function.py)), e.g. ```cifar10_test()```, which can be found in ```dataset_function.py```.

2. Check if your transform can be found in files under the dataset ```functional/generator_function/transforms```(click [here](functional/generator_function/transforms)), if not, you should first add the Transform class into the files under this directory, just follow the ```functional/generator_function/transforms/iaa_transform.py``` and it's very easy.

3. And then add the config to the ```configs/generate_config.py```(click [here](configs/generate_config.py)), which should be defined in format of list, and remember its name.

4. Run the command line. For example:
```shell
python generate.py --output_data_path dataset --base_dataset cifar10_test --store_name gaussian_blur --max_length 50000 --config iaa_gaussian_blur --cover
```

Then the file will be generated at ```dataset/gaussian_blur```, where you can see the ```description.txt``` to log the generation parameters

