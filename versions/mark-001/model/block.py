import random
import numpy as np

from model.point import Point


class Block:

    # UN USED VALUE
    COLOR_0 = (0, 0, 0)
    BLOCK_0 = [
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ]

    COLOR_1 = (57, 65, 86)
    BLOCK_1 = [
        [
            [1, 1],
            [1, 1],
        ],
        [
            [1, 1],
            [1, 1],
        ],
        [
            [1, 1],
            [1, 1],
        ],
        [
            [1, 1],
            [1, 1],
        ],
    ]

    COLOR_2 = (230, 41, 185)
    BLOCK_2 = [
        [
            [0, 1, 0],
            [1, 1, 1],
        ],
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
    ]

    COLOR_3 = (114, 203, 59)
    BLOCK_3 = [
        [
            [0, 1, 1],
            [1, 1, 0],
        ],
        [
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
        ],
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
    ]

    COLOR_4 = (255, 50, 19)
    BLOCK_4 = [
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ],
        [
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
        ],
    ]

    COLOR_5 = (255, 151, 28)
    BLOCK_5 = [
        [
            [1, 0, 0],
            [1, 1, 1],
        ],
        [
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
    ]

    COLOR_6 = (3, 65, 174)
    BLOCK_6 = [
        [
            [0, 0, 1],
            [1, 1, 1],
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
    ]

    COLOR_7 = (0, 210, 220)
    BLOCK_7 = [
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
    ]

    # Projection
    COLOR_8 = (255, 255, 255)
    BLOCK_8 = []

    BLOCKS = [BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3, BLOCK_4, BLOCK_5, BLOCK_6, BLOCK_7, BLOCK_8]
    COLORS = [COLOR_0, COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5, COLOR_6, COLOR_7, COLOR_8]

    def __init__(self, board_h, board_w, y=0):
        self.board_h = board_h
        self.board_w = board_w
        self.id = random.randint(1, 7)
        self.rotation = 0
        self.color = self.COLORS[self.id]
        self.value = np.array(self.BLOCKS[self.id][self.rotation]) * self.id
        self.point = Point((self.board_w - self.value.shape[1]) // 2, y)
        self.score = np.sum(self.get_identity())
        self.update()

    def update(self):
        self.shape = self.value.shape
        self.height, self.width = self.value.shape

    def rotate(self, clockwise=False):
        self.rotation += 1 if clockwise else -1
        
        if self.rotation >= 4:
            self.rotation = 0
        if self.rotation < 0:
            self.rotation = 3

        self.value = np.array(self.BLOCKS[self.id][self.rotation]) * self.id
        self.update()

    def get_identity(self):
        return np.where(self.value > 0, 1, 0).astype(np.int8)
