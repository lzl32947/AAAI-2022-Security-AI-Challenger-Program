# Files contains the config to generate the images

null = [
    [],
]

iaa_gaussian_blur = {
    "image_use": 1,
    "procedures":
        [
            ["IAAGaussianBlur", 0.8, {"sigma": (3, 5)}]
        ],
}

separate_gaussian_blur = {
    "image_use": 1,
    "procedures": [
        [
            ["IAAGaussianBlur", 1, {"sigma": (0, 0.6)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (0.6, 1.2)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (1.2, 1.8)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (1.8, 2.4)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (2.4, 3.0)}]
        ],
    ]
}
