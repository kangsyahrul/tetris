import numpy as np
#
# value = [[0, 0, 0], [0, 0, 6]]
# block_i = [[0, 1, 1], [1, 1, 0]]
#
# value = np.array(value)
# block_i = np.array(block_i)
#
# m1 = value.copy()
# m2 = block_i.copy()
#
# m1[m1 > 0] = 1
# m2[m2 > 0] = 1
#
# print(f'm1: {m1}')
# print(f'm2: {m2}')
#
# overlap = m1 + m2
# print(f'overlap: {overlap}')

arr = np.array([
    [1, 0],
    [2, 0],
    [1, 1],
])

for y in range(arr.shape[0]):
    a = arr[y, :] > 0
    print(False in a)

