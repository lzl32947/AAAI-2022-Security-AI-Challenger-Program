# AAAI-2022 Security AI Challenger Program

This is the repo for Aaai-2022 security AI Challenger program phase VIII.

## Base

The files in ```base``` directory is the original directory for baseline files, and can be found
in [training_template_for_AI_challenger_sea8](https://github.com/vtddggg/training_template_for_AI_challenger_sea8)

These files should not be modified to overwrite the logic and running procedures.

Original ```Readme``` can be found in [here](doc/README.md).

## Run baseline

The main script of the project is ```main.py```, and the parsed args can be found at ```util/tools/args_util.py```(
click [here](util/tools/args_util.py))

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

When training finished, all the files can be packed with the ```pack_upload.py```, which args can be found same at ```util/tools/args_util.py```(
click [here](util/tools/args_util.py))

Typically, can run the following commands:

```shell
python pack_upload.py --log_name cifar10 --identifier 20211128_213656 --data_dir dataset/baseline
```

Then you can find the packed zip file in ```upload``` directory.

## Data Storage
