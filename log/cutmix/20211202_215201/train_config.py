# This is the original config files from https://github.com/vtddggg/training_template_for_AI_challenger_sea8/blob/main/config.py
# Modifies should be commented

args_resnet = {
    'epochs': 160,  # 200 for original
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': 'CosineAnnealingLR',
    'scheduler_hyperparameters': {
        'T_max': 200
    },
    'batch_size': 192,  # 256 for original
}
args_densenet = {
    'epochs': 100,
    'optimizer_name': 'SGD',
    'optimizer_hyperparameters': {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'scheduler_name': None,
    'batch_size': 192,  # 256 for original
}
