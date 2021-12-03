import itertools
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


class Drawer:
    def __init__(self, **kwargs) -> None:
        self.figure = None
        if kwargs is not None:
            self.figure = plt.figure(
                num=kwargs["num"] if "num" in kwargs.keys() else None,
                figsize=kwargs["figsize"] if "figsize" in kwargs.keys() else None,
                dpi=kwargs["dpi"] if "dpi" in kwargs.keys() else None,
            )

    def show_image(self) -> None:
        """
        Show image.
        :return: None
        """
        if self.figure is None:
            raise RuntimeError("Figure is None!")
        else:
            self.figure.tight_layout()
            self.figure.show()

    def get_image(self) -> plt.Figure:
        """
        Show image.
        :return: None
        """
        if self.figure is None:
            raise RuntimeError("Figure is None!")
        else:
            self.figure.tight_layout()
            return self.figure

    def save_image(self, save_path: str) -> None:
        """
        Save image.
        :param save_path: str, the given path
        :return: None
        """
        if self.figure is None:
            raise RuntimeError("Figure is None!")
        else:
            self.figure.tight_layout()
            self.figure.savefig(save_path)

    def clear(self) -> None:
        """
        Clear images.
        :return: None
        """
        plt.close(self.figure)
        self.figure = None


class MatrixDrawer(Drawer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_batch(self, matrix, rows, current_row, **kwargs):
        # Convert tensor to ndarray
        if isinstance(matrix, torch.Tensor):
            if matrix.is_cuda:
                matrix = matrix.cpu()
            matrix = matrix.numpy()

        batch_size = matrix.shape[0]
        # Plot the matrix
        for index, item in enumerate(matrix):
            plot_positions = (rows, batch_size, (current_row - 1) * batch_size + index + 1)
            ax = self.figure.add_subplot(*plot_positions)
            mat = ax.imshow(item, aspect='auto', interpolation='nearest',
                            cmap=plt.cm.Blues if "cmap" not in kwargs.keys() else kwargs["cmap"])
            # Set title
            if "title" in kwargs.keys():
                ax.set_title("{}_{}".format(kwargs["title"], index + 1))

            self.figure.colorbar(mat, ax=ax)
            for i, j in itertools.product(range(item.shape[0]), range(item.shape[1])):
                plt.text(j, i, "{:<.4f}".format(item[i, j]),
                         horizontalalignment="center",
                         color="white" if item[i, j] > 0.5 else "black")

    def plot_matrix(self, matrix, **kwargs):

        # Convert tensor to ndarray
        if isinstance(matrix, torch.Tensor):
            if matrix.is_cuda:
                matrix = matrix.cpu()
            matrix = matrix.numpy()

        ax = self.figure.add_subplot(1, 1, 1)
        mat = ax.imshow(matrix, interpolation='nearest',
                        cmap=plt.cm.Blues if "cmap" not in kwargs.keys() else kwargs["cmap"])
        if "title" in kwargs.keys():
            ax.set_title(kwargs["title"])

        self.figure.colorbar(mat, ax=ax)
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            self.figure.text(j, i, "{:<.4f}".format(matrix[i, j]),
                             horizontalalignment="center",
                             color="white" if matrix[i, j] > 0.5 else "black")
        self.figure.tight_layout()


class ImageDrawer(Drawer):
    """
    Class for image drawing
    """

    @staticmethod
    def check_input(input_image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        # Convert tensor to ndarray
        if isinstance(input_image, torch.Tensor):
            if input_image.is_cuda:
                input_image = input_image.cpu()
            input_image = input_image.numpy()

        # Check value
        if np.max(input_image) > 255:
            raise RuntimeError("Pixel value should lower than 255!")
        if np.min(input_image) < -1:
            raise RuntimeError("Pixel value should bigger than -1!")

        # In range [0, 255]
        if np.max(input_image) > 1:
            if np.max(input_image) < 0:
                raise RuntimeError("Pixel value should be in range (0,255)")
            input_image = input_image.astype(np.uint8)

        # In range [0, 1]
        elif np.max(input_image) <= 1 and np.min(input_image) >= 0:
            input_image = input_image * 255
            input_image = input_image.astype(np.uint8)
        # In range [-1, 1]
        elif np.max(input_image) <= 1 and np.min(input_image) >= -1:
            input_image = input_image * 127.5
            input_image = input_image + 127.5
            input_image = input_image.astype(np.uint8)
        else:
            raise RuntimeError("Unknown image format!")
        return input_image

    def draw_batch(self, input_image: Union[np.ndarray, torch.Tensor], rows: int, current_row: int, **kwargs) -> None:
        # Init the figure if empty
        if self.figure is None:
            self.figure = plt.figure(
                num=kwargs["num"] if "num" in kwargs.keys() else None,
                figsize=kwargs["figsize"] if "figsize" in kwargs.keys() else None,
                dpi=kwargs["dpi"] if "dpi" in kwargs.keys() else None,
            )

        input_image = self.check_input(input_image)
        batch_size = None
        # Process the dimension
        if len(input_image.shape) == 2:
            batch_size = 1
            input_image = np.expand_dims(input_image, axis=0)
            input_image = np.concatenate([input_image, input_image, input_image], axis=0)
            input_image = np.expand_dims(input_image, axis=0)
        if len(input_image.shape) == 3:
            batch_size = input_image.shape[0]
            input_image = np.expand_dims(input_image, -1)
            input_image = np.concatenate([input_image, input_image, input_image], axis=-1)
        if len(input_image.shape) == 4:
            batch_size = input_image.shape[0]
            if input_image.shape[1] == 1:
                input_image = np.repeat(input_image, 3, axis=1)
            if input_image.shape[1] == 3:
                input_image = np.transpose(input_image, [0, 2, 3, 1])
            if input_image.shape[3] == 1:
                input_image = np.repeat(input_image, 3, axis=3)

        # Plot the image
        for index, item in enumerate(input_image):
            plot_positions = (rows, batch_size, (current_row - 1) * batch_size + index + 1)
            ax = self.figure.add_subplot(*plot_positions)
            ax.axis("off")
            ax.imshow(item)
            # Set title
            if kwargs["title"] is not None:
                ax.set_title("{}_{}".format(kwargs["title"][index], index + 1))

    def draw_same_batch(self, input_image: Union[np.ndarray, torch.Tensor], row: int, **kwargs) -> None:
        """
        Draw the images with the same batch
        :param input_image: np.ndarray or torch.Tensor
        :param row: int, the row to use in the figure
        :param kwargs: Dict, the packed parameters
        :return: None
        """
        items = len(input_image)
        col = items // row if items % row == 0 else items // row + 1
        count = 1
        for i in range(1, row + 1):
            for j in range(1, col + 1):
                if count > items:
                    break
                self.draw_image(input_image[count - 1], plot_position=(row, col, count), **kwargs)
                count += 1

    def draw_image(self, input_image: Union[np.ndarray, torch.Tensor], plot_position: tuple, **kwargs) -> None:
        """
        Draw single image or only part of the image.
        :param input_image: either be np.ndarray or torch.Tensor, the image to be plotted
        :param plot_position: tuple, the position to be placed in figure.add_subplot, like (1, 2, 1)
        :param kwargs: Dict, other useful information
        :return: None
        """
        # Init the figure if empty
        if self.figure is None:
            self.figure = plt.figure(
                num=kwargs["num"] if "num" in kwargs.keys() else None,
                figsize=kwargs["figsize"] if "figsize" in kwargs.keys() else None,
                dpi=kwargs["dpi"] if "dpi" in kwargs.keys() else None,
            )

        input_image = self.check_input(input_image)

        # Process the dimension
        if len(input_image.shape) == 2:
            input_image = np.concatenate([input_image, input_image, input_image], axis=0)
        if len(input_image.shape) == 4:
            if input_image.shape[0] != 1:
                raise RuntimeError("Multi images detected!")
            input_image = np.squeeze(input_image, 0)
        if len(input_image.shape) == 3:
            if input_image.shape[0] == 1:
                input_image = np.repeat(input_image, 3, axis=0)
            if input_image.shape[0] == 3:
                input_image = np.transpose(input_image, [1, 2, 0])
            if input_image.shape[2] == 1:
                input_image = np.repeat(input_image, 3, axis=2)

        # Plot the image
        ax = self.figure.add_subplot(*plot_position)
        ax.axis("off")
        ax.imshow(input_image)
        # Set title
        if "title" in kwargs.keys():
            if isinstance(kwargs["title"], str):
                ax.set_title(kwargs["title"])
            elif isinstance(kwargs["title"], list) or isinstance(kwargs["title"], tuple):
                ax.set_title(kwargs["title"][plot_position[-1] - 1])
