# Files contains the config to generate the images

null = [
    [],
]

iaa_gaussian_blur = {
    "config": {"use_images": 1},
    "procedures": [
        [
            ["IAAGaussianBlur", 0.8, {"sigma": (3, 5)}]
        ],
    ]
}

vertical_cutup_blur = {
    "config": {"use_images": 2},
    "procedures": [
        [
            ["VerticalCutup", 1, {}],
            ["IAAGaussianBlur", 1, {"sigma": (1, 1.6)}]
        ], [
            ["VerticalCutup", 1, {}],
            ["IAAGaussianBlur", 1, {"sigma": (1.6, 2.2)}]
        ], [
            ["VerticalCutup", 1, {}],
            ["IAAGaussianBlur", 1, {"sigma": (2.2, 2.8)}]
        ], [
            ["VerticalCutup", 1, {}],
            ["IAAGaussianBlur", 1, {"sigma": (2.8, 3.4)}]
        ], [
            ["VerticalCutup", 1, {}],
            ["IAAGaussianBlur", 1, {"sigma": (3.4, 4)}]
        ]
    ]
}

separate_gaussian_blur = {
    "config": {"use_images": 1},
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


separate_gaussian_blur_upper_2 = {
    "config": {"use_images": 1},
    "procedures": [
        [
            ["IAAGaussianBlur", 1, {"sigma": (1, 1.4)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (1.4, 1.8)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (1.8, 2.2)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (2.2, 2.6)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (2.6, 3.0)}]
        ],
    ]
}

separate_gaussian_blur_upper = {
    "config": {"use_images": 1},
    "procedures": [
        [
            ["IAAGaussianBlur", 1, {"sigma": (1, 1.6)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (1.6, 2.2)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (2.2, 2.8)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (2.8, 3.4)}]
        ],
        [
            ["IAAGaussianBlur", 1, {"sigma": (3.4, 4.0)}]
        ],
    ]
}
