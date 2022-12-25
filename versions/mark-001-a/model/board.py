import cv2
import random
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

    def __init__(self, WINDOW_SIZE=(288, 528), PADDING=(24, 24), BOARD=(10, 20), BLOCK_SIZE=(24, 24)):
        # Game Settings
        self.window_w, self.window_h = WINDOW_SIZE
        self.padding_w, self.padding_h = PADDING
        self.board_w, self.board_h = BOARD
        self.block_w, self.block_h = BLOCK_SIZE

        # User Interface
        self.background = self.create_background()

        # Start new game
        self.restart()
    
    # Game State
    def restart(self):
        self.is_game_over = False
        self.score = 0
        self.total_reward = 0
        self.next_blocks = [Block(self.board_h, self.board_w) for i in range(3)]
        self.new_block()
        self.value = self.create_board()
        self.state = self.get_state()

    def get_state(self):
        # Board State
        # state = self.put_block(self.block, self.block.point).flatten()
        # state = np.where(state > 0, 1, 0).astype(np.int32)

        # Holes, height, 
        projection_point = self.get_projection_point(self.block, self.block.point)
        state = self.put_block(self.block, projection_point)
        state = np.where(state > 0, 1, 0).astype(np.int32)

        holes = 20 - self.count_holes(state).flatten()
        heights = self.measure_heights(state).flatten()
        id = self.block.id
        rotation = self.block.rotation
        x, y = self.block.point.x, self.block.point.y
        next_ids = [block.id for block in self.next_blocks]

        state = []
        state.extend(list(holes))
        state.extend(list(heights))
        state.extend([id * 3 - 2, rotation * 5 + 5, x * 2, y])
        # state.extend(next_ids)
        state = np.array(state, dtype=np.int32)
        state[state > 20] = 20
        state[state < -1] = -1
        # state = np.vstack([holes, heights, id, rotation, x, y, next_ids]).flatten()
        # state = np.where(state > 20, 20, 0).astype(np.int32)

        return state
    
    def new_block(self):
        self.total_action = 0
        self.block = self.next_blocks[0]
        self.next_blocks.pop(0)
        self.next_blocks.append(Block(self.board_h, self.board_w))

    # User Interface
    def create_board(self):
        self.value = np.zeros((self.board_h, self.board_w), dtype=np.int32)

        for w in range(self.board_w):
            for h in range(random.randint(0, 2)):
                self.value[self.board_h - 1 - h, w] = 1
        
        
        while True:
            lines_removed = self.remove_lines()
            if lines_removed == 0:
                break
        
        return self.value

    def create_background(self, show_line=False):
        # Background
        background = np.zeros((self.window_h, self.window_w, 3), dtype=np.int32)

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

    def render(self):
        values = self.put_block(self.block, self.block.point)
        point_projection = self.get_projection_point(self.block, self.block.point)
        values = self.put_block(self.block, point_projection, values_projection=values)
        img_board = self.draw_board(value_temp=values.reshape((20, 10))).astype(np.uint8)
        
        # Score
        img_score = np.zeros_like(img_board)
        img_score = cv2.putText(img_score, f'Score: {self.score:.02f}', (self.padding_w, self.padding_h * 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        img_score = cv2.putText(img_score, f'Reward: {self.total_reward:.02f}', (self.padding_w, self.padding_h * 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        img_score = cv2.putText(img_score, f'Next Block:', (self.padding_w, self.padding_h * 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        for i, block in enumerate(self.next_blocks):
            img = np.zeros((4 * self.block_h, 5 * self.block_w, 3), dtype=np.int8)
            for y, row in enumerate(block.value):
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
            
            img_score[
                self.padding_h * 5 + i * 4 * self.board_h:self.padding_h * 5 + i * 4 * self.board_h + img.shape[0], 
                self.padding_w * 1:self.padding_w * 1 + img.shape[1]
            ] = img

        return cv2.hconcat([img_board, img_score])
    
    # Game Control
    def take_action(self, action):
        self.total_action += 1

        # Reward
        reward = 0.01

        # Take an action
        if action == 0:
            # Move left
            p = Point(self.block.point.x, self.block.point.y)
            if not self.is_game_over:
                p.move_left()
                if not self.is_block_overlap(self.block, p, verbose=True):
                    self.block.point.move_left()

        if action == 1:
            # Move right
            p = Point(self.block.point.x, self.block.point.y)
            if not self.is_game_over:
                p.move_right()
                if not self.is_block_overlap(self.block, p, verbose=True):
                    self.block.point.move_right()

        if action == 2:
            # Rotate
            if not self.is_game_over:
                self.block.rotate(clockwise=True)
                # print(f'block is rotated clockwise')
                p = Point(self.block.point.x, self.block.point.y)
                if self.is_block_overlap(self.block, p, verbose=True):
                    # print(f'block is overlap, trying to move left')
                    # try to move left 
                    p.move_left()
                    # print(f'block is moved left')
                    if self.is_block_overlap(self.block, p, verbose=True):
                        # print(f'block is still overlap, trying to move left again')
                        # try to move left 
                        p.move_left()
                        # print(f'block is moved left again')
                        if self.is_block_overlap(self.block, p, verbose=True):
                            # print(f'block is still overlap, restore to original position')
                            # rotate and left is still overlap, return to original 
                            p.move_right()
                            p.move_right()
                            # print(f'block is restored to original, trying to move fight')

                            # try to move right 
                            p.move_right()
                            # print(f'block is moved right')
                            if self.is_block_overlap(self.block, p, verbose=True):
                                # print(f'block is still overlap, trying to move right again')
                                # try to move left 
                                p.move_right()
                                # print(f'block is moved right again')
                                if self.is_block_overlap(self.block, p, verbose=True):
                                    # print(f'block is still overlap, restore to original position')
                                    # rotate and right is still overlap, return to original 
                                    p.move_left()
                                    p.move_left()
                                    # print(f'block is restored to original, game over!')
                                    self.block.rotate(clockwise=False)
                                    
                                    self.is_game_over = True
                # print()
                self.block.point = Point(p.x, p.y)

        
        if action == 3:
            reward = self.submit()
            self.total_action = 0

        elif self.total_action > 5:
            reward *= (5 / (self.total_action))

            if self.total_action > 15:
                reward = 0
                self.total_action = 0
                self.is_game_over = True

        self.state = self.get_state()
        self.is_game_over = self.check_game_over()

        self.total_reward += reward

        return self.is_game_over, reward, self.state

    # Game Logic
    def is_block_overlap(self, block, point, verbose=False):
        values = self.value.copy()

        # Check if block is overlap
        h_s, h_e, w_s, w_e = point.y, min(self.board_h, point.y + block.height), max(0, point.x), min(values.shape[1], point.x + block.width)
        value = np.copy(values[h_s:h_e, w_s:w_e])
        block_i = np.copy(block.get_identity())

        # Is overlap with the edge
        if value.shape != block_i.shape:
            if (max(0, point.x) == 0) and (np.sum(block_i[:, :(-point.x)]) == 0):
                return is_overlap(block_i[:, (-point.x):], value)

            if (min(values.shape[1], point.x + block.width) == values.shape[1]) and (np.sum(block_i[:, -(point.x + block.width - values.shape[1]):]) == 0):
                return is_overlap(block_i[:, :-(point.x + block.width - values.shape[1])], value)

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
        
        return values

    def remove_lines(self, y=0):
        # Remove all filled rows
        total_lines = 0
        y = self.board_h - 1
        while y >= 0:
            is_filled_up = not (False in list(self.value[y, :] > 0))
            if is_filled_up:
                # Remove line
                self.value[1:y + 1, :] = self.value[0:y, :]
                self.value[0, :] = np.zeros(self.board_w, dtype=np.int32)
                total_lines += 1

            else:
                y -= 1
        
        return total_lines

    def submit(self):
        params = {
            'value_before': [],
            'value_after': [],
            'block_height': 0,
            'block_score': 0,
            'lines_removed': 0
        }
        
        # Add score
        params['value_before'] = self.value.copy()
        params['block_height'] = self.block.point.y
        params['block_score']  = self.block.score

        projection_point = self.get_projection_point(self.block, self.block.point)
        w_s, w_e = max(0, projection_point.x), min(self.board_w, projection_point.x + self.block.width)
        h_s, h_e = max(0, projection_point.y), min(self.board_h, projection_point.y + self.block.height)
        
        # Not overlap with the projection
        if (w_e - w_s) != self.block.width:
            # print('Overlap with side')
            # print((h_s, h_e, w_s, w_e, self.block_identity, len(Block.BLOCKS) - 1, ))

            if w_e <= projection_point.x + self.block.width - 1:
                # print(f'{w_e} == {projection_point.x} + {self.block.width} - 1')
                self.value[h_s:h_e, w_s:w_e] += self.block.value[:, :(self.block.width - (projection_point.x + self.block.width - w_e))]
                
            if w_s == 0:
                # print(f'{w_s}')
                self.value[h_s:h_e, w_s:w_e] += self.block.value[:, (self.block.width - (w_e - w_s)):]
                
        else:
            self.value[h_s:h_e, w_s:w_e] += self.block.value
            # self.value[projection_point.y:projection_point.y+self.block.height, projection_point.x:projection_point.x+self.block.width] += self.block.value

        lines_removed = self.remove_lines()

        params['value_after'] = self.value.copy()
        params['lines_removed'] = lines_removed

        # Create new block
        self.new_block()

        # Collect score and rewards
        score, reward = self.reward_functions(params)
        self.score += score

        return reward

    def reward_functions(self, params):
        value_before  = params['value_before']
        value_after   = params['value_after']
        block_height  = params['block_height']
        block_score   = params['block_score']
        lines_removed = params['lines_removed']

        reward = 0

        # Reward from block
        reward_block = 1
        reward += reward_block

        # Reward from lines removed
        reward_lines = lines_removed ** 2
        reward *= 1 + max(0, reward_lines * 10)
        
        # Holes punishment/reward
        holes_before = self.count_holes(value_before)
        holes_after = self.count_holes(value_after)
        total_holes_before = np.sum(holes_before)
        total_holes_after = np.sum(holes_after)
        reward_holes = (total_holes_after - total_holes_before) / 2
        reward *= 1 - min(0.9, reward_holes)
        
        # Height punishment/reward
        heights_before = np.max(self.measure_heights(value_before))
        heights_after = np.max(self.measure_heights(value_after))
        reward *= 1 - min(0.9, (heights_after - heights_before) / 4)
        
        return block_score, max(0.02, reward)
    
    def get_projection_point(self, block, point):
        p = Point(point.x, point.y)
        while p.y + block.height < self.value.shape[0]:
            p.y += 1
            if self.is_block_overlap(block, p, verbose=False):
                p.y -= 1
                break
        return p

    def check_game_over(self):
        # Check if game is over
        projection_point = self.get_projection_point(self.block, self.block.point)
        if projection_point == self.block.point:
            self.is_game_over = True
        
        holes = self.count_holes(self.value)
        if np.max(holes) >= 5:
            self.is_game_over = True

        return self.is_game_over
    
    # Helper Functions
    def count_holes(self, state):
        holes = [0] * state.shape[1]
        for w in range(state.shape[1]):
            flag = False
            count = 0
            for h in range(state.shape[0]):
                if state[h, w] > 0:
                    flag = True
                else:
                    if flag:
                        count += 1
            holes[w] = count

        return np.array(holes)
    
    def measure_heights(self, state):
        heights = [0] * state.shape[1]
        for w in range(state.shape[1]):
            for h in range(state.shape[0]):
                if state[h, w] > 0:
                    heights[w] = state.shape[0] - h
                    break
        return np.array(heights)
