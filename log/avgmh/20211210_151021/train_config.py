# This is the original config files from https://github.com/vtddggg/training_template_for_AI_challenger_sea8/blob/main/config.py
# Modifies should be commented

args_resnet = {
    'epochs': 200,  # 200 for original
    'optimizer_name': 'AdamW',
    'optimizer_hyperparameters': {
        'lr': 0.001,
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 32,  # 256 for original
}
args_densenet = {
    'epochs': 200,
    'optimizer_name': 'AdamW',
    'optimizer_hyperparameters': {
        'lr': 0.001,
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 32,  # 256 for original
}
