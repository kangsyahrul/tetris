import cv2
import numpy as np
from model.block import Block
from model.point import Point


def is_overlap(m1, m2):
    m1[m1 > 0] = 1
    m2[m2 > 0] = 1

    overlap = m1 + m2
    overlap[overlap < 2] = 0
    overlap[overlap > 1] = 1

    if np.sum(overlap) > 0:
        return True

    return False


class Board:

    def __init__(self, window_size, padding, board, block_size):
        self.window_w, self.window_h = window_size
        self.padding_w, self.padding_h = padding
        self.board_w, self.board_h = board
        self.block_w, self.block_h = block_size

        self.background = self.create_background()
        self.value = self.create_board()
        self.shape = self.value.shape

    def create_board(self):
        return np.array([[0 for x in range(self.board_w)] for y in range(self.board_h)], dtype=np.int8)

    def create_background(self):
        # Background
        background = np.zeros((self.window_h, self.window_w, 3), dtype=np.int8)

        # Border
        x1, y1 = self.padding_w, self.padding_h
        x2, y2 = self.window_w - self.padding_w, self.window_h - self.padding_h
        background = cv2.rectangle(background, (x1, y1), (x2, y2), (0, 0, 255), 4)

        # # Line Separator
        # for i in range(self.board_w):
        #     x = self.padding_w + self.block_w * i
        #     background = cv2.line(background, (x, y1), (x, y2), (0, 0, 255), 1)
        #
        # for i in range(self.board_h):
        #     y = self.padding_h + self.block_h * i
        #     background = cv2.line(background, (x1, y), (x2, y), (0, 0, 255), 1)

        return background

    def draw_board(self, value_temp=None):
        values = self.value if value_temp is None else value_temp

        img = self.background.copy()
        for y, row in enumerate(values):
            for x, id in enumerate(row):
                if id != 0:
                    color_block = Block.COLORS[id]
                    color_edge = (0, 0, 0)
                    if id == len(Block.BLOCKS) - 1:
                        color_temp = color_edge
                        color_edge = color_block
                        color_block = color_temp
                    x1, y1 = self.padding_w + self.block_w * x, self.padding_h + self.block_h * y
                    x2, y2 = x1 + self.block_w, y1 + self.block_h
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), color_block, -1)
                    img = cv2.rectangle(img, (x1+2, y1+2), (x2-2, y2-2), color_edge,  2)

        return img

    def is_block_overlap(self, block, point):
        values = self.value.copy()

        # Check if block is overlap
        value = np.copy(values[point.y:point.y+block.height, point.x:point.x+block.width])
        block_i = np.copy(block.get_identity())

        if value.shape != block_i.shape:
            return True

        return is_overlap(value, block_i)

    def put_block(self, block, point, values_projection=None):
        values = self.value.copy()
        if values_projection is not None:
            values = values_projection

        # Can not go outside the frame
        if point.x < 0:
            point.x = 0
        if point.y < 0:
            point.y = 0

        if point.x > self.board_w - block.width:
            point.x = self.board_w - block.width
        if point.y > self.board_h - block.height:
            point.y = self.board_h - block.height

        values_id = np.copy(values[point.y:point.y+block.height, point.x:point.x+block.width])
        values_id[values_id > 0] = 1
        is_overlap_ = is_overlap(values_id, block.get_identity())
        if values_projection is not None:
            values[point.y:point.y + block.height, point.x:point.x + block.width]\
                += block.get_identity() * (len(Block.BLOCKS) - 1)
        else:
            if not is_overlap_:
                values[point.y:point.y+block.height, point.x:point.x+block.width] += block.value
            else:
                # Stuck
                return None, point

        # Do not overlap with the projection
        values[values > len(Block.BLOCKS) - 1] -= len(Block.BLOCKS) - 1

        return values, point

    def remove_lines(self, y=0):
        # Remove all filled rows
        y = self.board_h - 1
        while y >= 0:
            arr = self.value[y, :] > 0
            is_filled_up = not (False in arr)
            if is_filled_up:
                # Move
                self.value[1:y+1, :] = self.value[0:y, :]
                self.value[0, :] = np.zeros(self.board_w, dtype=np.int8)
            else:
                y -= 1

    def submit(self, block, point):
        self.value[point.y:point.y+block.height, point.x:point.x+block.width] += block.value
        self.remove_lines()

    def get_projection_point(self, block, point):
        p = Point(point.x, point.y)
        overlap = False
        while not overlap:
            p.y += 1
            if self.is_block_overlap(block, p):
                p.y -= 1
                overlap = True
        return p
