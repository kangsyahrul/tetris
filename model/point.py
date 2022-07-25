class Point:
    x, y = 0, 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move_left(self):
        self.x = self.x - 1

    def move_right(self):
        self.x = self.x + 1

    def move_up(self):
        self.y = self.y - 1

    def move_down(self):
        self.y = self.y + 1

    def __copy__(self):
        return self

    def __str__(self):
        return f'({self.x}, {self.y})'
