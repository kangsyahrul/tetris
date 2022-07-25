import os
import sys
import cv2
import time
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

    block = Block()
    point = Point(board.board_w//2-1, 0)

    is_game_over = False
    is_stuck = False
    time_start = datetime.now()
    time_past = datetime.now()

    score = 0
    while True:
        sc.clear_screen()
        print(f'Score: {score}')
        print(f'Time: {str(datetime.now() - time_start).split(".")[0]}')

        if is_stuck:
            if point.y == 0:
                # Game Over
                is_game_over = True
                break
            score += block.score
            is_stuck = False
            board.submit(block, point)
            point = Point(board.board_w // 2 - 1, 0)
            block = Block()


        # Put the block
        values, point = board.put_block(block, point)
        if values is None:
            # Stuck
            is_stuck = True
            continue

        point_projection = board.get_projection_point(block, point)
        values, _ = board.put_block(block, point_projection, values_projection=values)
        if values is None:
            # Stuck
            print('Stuck')
            is_stuck = True
            continue

        img = board.draw_board(value_temp=values)

        # # Draw "point" coordinate
        # xc, yc = PADDING_X + point.x * BLOCK_SIZE_W + BLOCK_SIZE_W//2, PADDING_Y + point.y * BLOCK_SIZE_H + BLOCK_SIZE_H//2
        # img = cv2.circle(img, (xc, yc), BLOCK_SIZE_W//4, (255, 0, 0), -1)
        #
        # # Draw "block shape"
        # x1, y1 = PADDING_X + point.x * BLOCK_SIZE_W, PADDING_Y + point.y * BLOCK_SIZE_H
        # x2, y2 = PADDING_X + (point.x + block.width) * BLOCK_SIZE_W, PADDING_Y + (point.y + block.height) * BLOCK_SIZE_H
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Tetris', img)

        while True:
            key = cv2.waitKey(10)
            if key != -1:
                break
            if (datetime.now() - time_past).total_seconds() > 1:
                time_past = datetime.now()
                point.move_down()
                if board.is_block_overlap(block, point):
                    point.move_up()
                    is_stuck = True
                break

        if is_stuck:
            continue

        if key == ord('q'):
            break

        # Change block type
        if key == ord(' ') and not is_game_over:
            # Drop
            point = board.get_projection_point(block, point)
            is_stuck = True
            continue

        # Rotate: Clockwise
        if key == ord('r') and not is_game_over:
            # Check if rotating causes overlap
            block.rotate(clockwise=True)
            if board.is_block_overlap(block, point):
                print('Overlap: should be game over!')
                block.rotate(clockwise=False)

        # Rotate: CounterClockwise
        if key == ord('e') and not is_game_over:
            block.rotate(clockwise=False)
            if board.is_block_overlap(block, point):
                print('Overlap: should be game over!')
                block.rotate(clockwise=True)

        # Arrow: left
        p = Point(point.x, point.y)
        if (key == 81 or key == ord('a')) and not is_game_over:
            p.move_left()
            if not board.is_block_overlap(block, p):
                point.move_left()

        # # Arrow: up
        # if key == 82 and not is_game_over:
        #     p.move_up()
        #     if not board.is_block_overlap(block, p):
        #         point.move_up()

        # Arrow: right
        if (key == 83 or key == ord('d')) and not is_game_over:
            p.move_right()
            if not board.is_block_overlap(block, p):
                point.move_right()

        # Arrow: bottom
        if (key == 84 or key == ord('s')) and not is_game_over:
            p.move_down()
            if not board.is_block_overlap(block, p):
                point.move_down()

    if is_game_over:
        print('Game Over')
    else:
        print('Exiting the game')
    sys.exit(0)


if __name__ == '__main__':
    main()
