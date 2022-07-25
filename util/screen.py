import os


def clear_screen():
    # os.system('clear')
    os.system('cls' if os.name == 'nt' else 'clear')

