import os
import sys
import cv2
import time
import random

from datetime import datetime

import util.screen as sc
from model.board import Board
from model.block import Block
from model.point import Point

# GAME SETTING
BOARD_SIZE_W, BOARD_SIZE_H = BOARD = (10, 20)

# SCREEN SETTING
BLOCK_SIZE_W, BLOCK_SIZE_H = BLOCK_SIZE = (24, 24)
PADDING_X, PADDING_Y = PADDING = (24, 24)
WINDOW_SIZE_W, WINDOW_SIZE_H = WINDOW_SIZE = (PADDING_X * 2 + BOARD_SIZE_W * BLOCK_SIZE_W, PADDING_Y * 2 + BOARD_SIZE_H * BLOCK_SIZE_H)


def main():
    global WINDOW_SIZE, PADDING, BOARD
    sc.clear_screen()

    board = Board(WINDOW_SIZE, PADDING, BOARD, BLOCK_SIZE)

    block = Block(board.board_w)

    is_game_over = False
    is_stuck = False
    time_start = datetime.now()
    time_past = datetime.now()

    score = 0
    while True:
        # sc.clear_screen()
        # print(f'Score: {score}')
        # print(f'Time: {str(datetime.now() - time_start).split(".")[0]}')

        if is_stuck:
            if block.point.y == 0:
                # Game Over
                is_game_over = True
                break
            
            score += block.score
            is_stuck = False
            board.submit(block, block.point)
            block = Block(board.board_w)


        # Put the block
        values = board.put_block(block, block.point)
        if values is None:
            # Stuck
            is_stuck = True
            continue

        point_projection = board.get_projection_point(block, block.point)
        values = board.put_block(block, point_projection, values_projection=values)
        if values is None:
            # Stuck
            print('Stuck')
            is_stuck = True
            continue

        img = board.draw_board(value_temp=values)

        # # Draw "point" coordinate
        # xc, yc = PADDING_X + block.point.x * BLOCK_SIZE_W + BLOCK_SIZE_W//2, PADDING_Y + block.point.y * BLOCK_SIZE_H + BLOCK_SIZE_H//2
        # img = cv2.circle(img, (xc, yc), BLOCK_SIZE_W//4, (255, 0, 0), -1)
        #
        # # Draw "block shape"
        # x1, y1 = PADDING_X + block.point.x * BLOCK_SIZE_W, PADDING_Y + block.point.y * BLOCK_SIZE_H
        # x2, y2 = PADDING_X + (block.point.x + block.width) * BLOCK_SIZE_W, PADDING_Y + (block.point.y + block.height) * BLOCK_SIZE_H
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Tetris', img)

        # Gravity
        while True:
            key = cv2.waitKey(10)
            if key != -1:
                break
                
            if (datetime.now() - time_past).total_seconds() > 1:
                time_past = datetime.now()
                block.point.move_down()
                if board.is_block_overlap(block, block.point):
                    block.point.move_up()
                    is_stuck = True
                break

        if is_stuck:
            continue

        if key == ord('q'):
            break

        # Drop the block
        if key == ord(' ') and not is_game_over:
            # Drop
            block.point = point_projection
            is_stuck = True

            # # Only change the block (DEBUG)
            # block = Block(board.board_w)
            continue

        # Rotate: Clockwise
        if key == ord('r') and not is_game_over:
            # Check if rotating causes overlap
            block.rotate(clockwise=True)
            # print(f'block is rotated clockwise')
            p = Point(block.point.x, block.point.y)
            if board.is_block_overlap(block, p, verbose=True):
                # print(f'block is overlap, trying to move left')
                # try to move left 
                p.move_left()
                # print(f'block is moved left')
                if board.is_block_overlap(block, p, verbose=True):
                    # print(f'block is still overlap, trying to move left again')
                    # try to move left 
                    p.move_left()
                    # print(f'block is moved left again')
                    if board.is_block_overlap(block, p, verbose=True):
                        # print(f'block is still overlap, restore to original position')
                        # rotate and left is still overlap, return to original 
                        p.move_right()
                        p.move_right()
                        # print(f'block is restored to original, trying to move fight')

                        # try to move right 
                        p.move_right()
                        # print(f'block is moved right')
                        if board.is_block_overlap(block, p, verbose=True):
                            # print(f'block is still overlap, trying to move right again')
                            # try to move left 
                            p.move_right()
                            # print(f'block is moved right again')
                            if board.is_block_overlap(block, p, verbose=True):
                                # print(f'block is still overlap, restore to original position')
                                # rotate and right is still overlap, return to original 
                                p.move_left()
                                p.move_left()
                                # print(f'block is restored to original, game over!')
                                block.rotate(clockwise=False)
            # print()
            block.point = Point(p.x, p.y)

        # Rotate: CounterClockwise
        if key == ord('e') and not is_game_over:
            block.rotate(clockwise=False)
            if board.is_block_overlap(block, block.point, verbose=True):
                print('Overlap: should be game over!')
                block.rotate(clockwise=True)

        # Arrow: left
        p = Point(block.point.x, block.point.y)
        if (key == 81 or key == ord('a')) and not is_game_over:
            p.move_left()

            if not board.is_block_overlap(block, p, verbose=True):
                block.point.move_left()
            else:
                print('Cannot move left!')

        # Arrow: right
        if (key == 83 or key == ord('d')) and not is_game_over:
            p.move_right()
            if not board.is_block_overlap(block, p, verbose=True):
                block.point.move_right()
            else:
                print('Cannot move right!')

        # Arrow: up
        if (key == 82 or key == ord('w')) and not is_game_over:
            p.move_up()
            if not board.is_block_overlap(block, p, verbose=True):
                block.point.move_up()

        # Arrow: bottom
        if (key == 84 or key == ord('s')) and not is_game_over:
            p.move_down()
            if not board.is_block_overlap(block, p, verbose=True):
                block.point.move_down()

    if is_game_over:
        print('Game Over')
    else:
        print('Exiting the game')
    
    sys.exit(0)


if __name__ == '__main__':
    main()
