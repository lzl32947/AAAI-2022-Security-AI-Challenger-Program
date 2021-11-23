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
python main.py --log_name base --data_train dataset/baseline --data_eval dataset/baseline --output_checkpoint checkpoint
```