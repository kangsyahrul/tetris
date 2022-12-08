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

    REWARD = 10

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

    def create_background(self, show_line=False):
        # Background
        background = np.zeros((self.window_h, self.window_w, 3), dtype=np.int8)

        # Border
        x1, y1 = self.padding_w, self.padding_h
        x2, y2 = self.window_w - self.padding_w, self.window_h - self.padding_h
        background = cv2.rectangle(background, (x1, y1), (x2, y2), (0, 0, 255), 4)

        if show_line:
            # Line Separator
            for i in range(self.board_w):
                x = self.padding_w + self.block_w * i
                background = cv2.line(background, (x, y1), (x, y2), (0, 0, 255), 1)
            
            for i in range(self.board_h):
                y = self.padding_h + self.block_h * i
                background = cv2.line(background, (x1, y), (x2, y), (0, 0, 255), 1)

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

    def is_block_overlap(self, block, point, verbose=False):
        values = self.value.copy()

        # Check if block is overlap
        h_s, h_e, w_s, w_e = point.y, min(self.board_h, point.y + block.height), max(0, point.x), min(values.shape[1], point.x + block.width)
        value = np.copy(values[h_s:h_e, w_s:w_e])
        block_i = np.copy(block.get_identity())

        # Is overlap with the edge
        if value.shape != block_i.shape:
            if max(0, point.x) == 0:
                if verbose: print('Block is on the left of the edge!')
                if (np.sum(block_i[:, :(-point.x)]) == 0) and (np.sum(value[:, (-point.x):]) == 0):
                    # if verbose: print('but its oke')
                    return False
                else:
                    # if verbose: print('but its not oke')
                    pass

            if min(values.shape[1], point.x + block.width) == values.shape[1]:
                if verbose: print('Block is on the right of the edge!')
                if (np.sum(block_i[:, -(point.x + block.width - values.shape[1]):]) == 0) and (np.sum(value[:, :-(point.x + block.width - values.shape[1])]) == 0): # and is_overlap(value, block_i[:, :-1]):
                    # if verbose: print('but its oke')
                    return False
                else:
                    # if verbose: print('but its not oke')
                    pass
                
            return True

        return is_overlap(value, block_i)

    def put_block(self, block, point, values_projection=None):
        values = self.value.copy()
        if values_projection is not None:
            values = values_projection
            color = 8
        else:
            color = block.id

        w_s, w_e = max(0, point.x), min(self.board_w, point.x + block.width)
        h_s, h_e = max(0, point.y), min(self.board_h, point.y + block.height)
        block_identity = block.get_identity()

        # Not overlap with the projection
        if (w_e - w_s) != block.width:
            # print('Overlap with side')
            # print((h_s, h_e, w_s, w_e, block_identity, len(Block.BLOCKS) - 1, ))

            if w_e <= point.x + block.width - 1:
                # print(f'{w_e} == {point.x} + {block.width} - 1')
                values[h_s:h_e, w_s:w_e] += (block_identity * color)[:, :(block.width - (point.x + block.width - w_e))]
                
            if w_s == 0:
                # print(f'{w_s}')
                values[h_s:h_e, w_s:w_e] += (block_identity * color)[:, (block.width - (w_e - w_s)):]
                
        else:
            values[h_s:h_e, w_s:w_e] += block_identity * color

        values[values > len(Block.BLOCKS) - 1] = len(Block.BLOCKS) - 1
        # values[values > len(Block.BLOCKS) - 1] -= len(Block.BLOCKS) - 1


        # if not is_projection:
        #     self.value = values

        # # Can not go outside the frame
        # if point.x < -1:
        #     point.x = -1
        # if point.y < 0:
        #     point.y = 0

        # if point.x > self.board_w - block.width + 1:
        #     point.x = self.board_w - block.width + 1
        # if point.y > self.board_h - block.height:
        #     point.y = self.board_h - block.height

        # values_id = np.copy(values[point.y:point.y+block.height, point.x:point.x+block.width])
        # values_id[values_id > 0] = 1
        # if values_projection is not None:
        #     if max(-1, point.x) == -1 or min(values.shape[1] + 1, point.x+block.width) == values.shape[1] + 1:
        #         if max(-1, point.x) == -1:
        #             if np.sum(block.value[:, 0]) == 0:
        #                 values[point.y:point.y+block.height, max(0, point.x):min(values.shape[1]+1, point.x+block.width)] += (block.get_identity() * (len(Block.BLOCKS) - 1))[:, 1:]

        #         if min(values.shape[1] + 1, point.x+block.width) == values.shape[1]+1:
        #             if np.sum(block.value[:, -1]) == 0:
        #                 values[point.y:point.y+block.height, max(-1, point.x):min(values.shape[1]+1, point.x+block.width)] += (block.get_identity() * (len(Block.BLOCKS) - 1))[:, :-1]
        #     else:
        #         values[point.y:point.y+block.height, point.x:point.x + block.width] += block.get_identity() * (len(Block.BLOCKS) - 1)

        # else:
        #     is_overlap_ = False # is_overlap(values_id, block.get_identity())
            
        #     if max(-1, point.x) == -1 or min(values.shape[1] + 1, point.x+block.width) == values.shape[1] + 1:
        #         if max(-1, point.x) == -1:
        #             if np.sum(block.value[:, 0]) == 0:
        #                 values[point.y:point.y+block.height, max(0, point.x):min(values.shape[1]+1, point.x+block.width)] += block.value[:, 1:]

        #         if min(values.shape[1] + 1, point.x+block.width) == values.shape[1]+1:
        #             if np.sum(block.value[:, -1]) == 0:
        #                 values[point.y:point.y+block.height, max(-1, point.x):min(values.shape[1]+1, point.x+block.width)] += block.value[:, :-1]

        #     else:            
        #         if not is_overlap_:
        #             values[point.y:point.y+block.height, point.x:point.x+block.width] += block.value
        #         else:
        #             # Stuck
        #             return None, point

        # # Not overlap with the projection
        # values[values > len(Block.BLOCKS) - 1] -= len(Block.BLOCKS) - 1

        return values

    def remove_lines(self, y=0):
        reward = 0
        # Remove all filled rows
        y = self.board_h - 1
        while y >= 0:
            arr = self.value[y, :] > 0
            is_filled_up = not (False in arr)
            if is_filled_up:
                # Move
                self.value[1:y+1, :] = self.value[0:y, :]
                self.value[0, :] = np.zeros(self.board_w, dtype=np.int8)
                reward += self.REWARD

            else:
                y -= 1
        
        return reward

    def submit(self, block, point):
        
        w_s, w_e = max(0, point.x), min(self.board_w, point.x + block.width)
        h_s, h_e = max(0, point.y), min(self.board_h, point.y + block.height)
        
        # Not overlap with the projection
        if (w_e - w_s) != block.width:
            # print('Overlap with side')
            # print((h_s, h_e, w_s, w_e, block_identity, len(Block.BLOCKS) - 1, ))

            if w_e <= point.x + block.width - 1:
                # print(f'{w_e} == {point.x} + {block.width} - 1')
                self.value[h_s:h_e, w_s:w_e] += block.value[:, :(block.width - (point.x + block.width - w_e))]
                
            if w_s == 0:
                # print(f'{w_s}')
                self.value[h_s:h_e, w_s:w_e] += block.value[:, (block.width - (w_e - w_s)):]
                
        else:
            self.value[h_s:h_e, w_s:w_e] += block.value
            # self.value[point.y:point.y+block.height, point.x:point.x+block.width] += block.value

        return self.remove_lines()

    def get_projection_point(self, block, point):
        p = Point(point.x, point.y)
        while p.y + block.height < self.value.shape[0]:
            p.y += 1
            if self.is_block_overlap(block, p, verbose=False):
                p.y -= 1
                break
        return p
