import random
import numpy as np


class Block:

    COLOR_0 = (0, 0, 0)
    BLOCK_0 = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    COLOR_1 = (57, 65, 86)
    BLOCK_1 = [
        [1, 1],
        [1, 1],
    ]

    COLOR_2 = (230, 41, 185)
    BLOCK_2 = [
        [0, 1, 0],
        [1, 1, 1],
    ]

    COLOR_3 = (114, 203, 59)
    BLOCK_3 = [
        [0, 1, 1],
        [1, 1, 0],
    ]

    COLOR_4 = (255, 50, 19)
    BLOCK_4 = [
        [1, 1, 0],
        [0, 1, 1],
    ]

    COLOR_5 = (255, 151, 28)
    BLOCK_5 = [
        [1, 0, 0],
        [1, 1, 1],
    ]

    COLOR_6 = (3, 65, 174)
    BLOCK_6 = [
        [1, 1, 1],
        [1, 0, 0],
    ]

    COLOR_7 = (0, 210, 220)
    BLOCK_7 = [
        [1, 1, 1, 1],
    ]

    # Projection
    COLOR_8 = (255, 255, 255)
    BLOCK_8 = []

    BLOCKS = [BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3, BLOCK_4, BLOCK_5, BLOCK_6, BLOCK_7, BLOCK_8]
    COLORS = [COLOR_0, COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5, COLOR_6, COLOR_7, COLOR_8]

    def __init__(self):
        self.id = random.randint(1, 7)
        self.color = self.COLORS[self.id]
        self.value = np.array(self.BLOCKS[self.id]) * self.id
        self.score = sum([sum(arr) for arr in np.array(self.BLOCKS[self.id])])
        self.update()

    def update(self):
        self.shape = self.value.shape
        self.height = self.value.shape[0]
        self.width = self.value.shape[1]

    def rotate(self, clockwise=False):
        if clockwise:
            self.value = np.rot90(self.value, k=3, axes=(0, 1))
            self.update()
        else:
            self.value = np.rot90(self.value, k=1, axes=(0, 1))
            self.update()

    def get_identity(self):
        return np.array(self.value // self.id, dtype=np.int8)
