import cv2
import numpy as np

from model.board import Board

# GAME SETTING
BOARD_SIZE_W, BOARD_SIZE_H = BOARD = (10, 20)

# SCREEN SETTING
BLOCK_SIZE_W, BLOCK_SIZE_H = BLOCK_SIZE = (24, 24)
PADDING_X, PADDING_Y = PADDING = (24, 24)
WINDOW_SIZE_W, WINDOW_SIZE_H = WINDOW_SIZE = (PADDING_X * 2 + BOARD_SIZE_W * BLOCK_SIZE_W, PADDING_Y * 2 + BOARD_SIZE_H * BLOCK_SIZE_H)
board = Board(WINDOW_SIZE, PADDING, BOARD, BLOCK_SIZE)


is_game_over = False
while True:
    img = board.render()
    # print(img)
    # print(img.shape)
    # # cv2.imshow('Tetris', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imshow('Tetris', img)

    key = cv2.waitKey(0)
    if key == ord('a'):
        is_game_over, reward, state = board.take_action(0)
        # print(f'is_game_over, reward, state: {(is_game_over, reward, state)}')
    
    if key == ord('d'):
        is_game_over, reward, state = board.take_action(1)
        # print(f'is_game_over, reward, state: {(is_game_over, reward, state)}')
    
    if key == ord('r'):
        is_game_over, reward, state = board.take_action(2)
        # print(f'is_game_over, reward, state: {(is_game_over, reward, state)}')
    
    if key == ord(' '):
        is_game_over, reward, state = board.take_action(3)
        # print(f'is_game_over, reward, state: {(is_game_over, reward, state)}')

    if key == 27 or key == ord('q'):
        break

    if is_game_over:
        print(f'GAME OVER!')
        break

cv2.destroyAllWindows()
