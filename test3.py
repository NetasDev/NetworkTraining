import numpy as np
from time import perf_counter
from contextlib import suppress

weights = np.array(([ 4,-3, 2, 2, 2, 2,-3, 4],
                                [-3,-4,-1,-1,-1,-1,-4,-3],
                                [ 2,-1, 1, 0, 0, 1, -1,2],
                                [ 2,-1, 0, 1, 1, 0,-1, 2],
                                [ 2,-1, 0, 1, 1, 0,-1, 2],
                                [ 2,-1, 1, 0, 0, 1,-1, 2],
                                [-3,-4,-1,-1,-1,-1,-4,-3],  
                                [ 4,-3, 2, 2, 2, 2,-3, 4]))

weights =             np.array(([ 5,-3, 3, 3,-3, 5],
                                [-3,-4,-1,-1,-4,-3],
                                [ 3,-1, 1, 1,-1, 3],
                                [ 3,-1, 1, 1,-1, 3],
                                [-3,-4,-1,-1,-4,-3],  
                                [ 5,-3, 3, 3,-3, 5]))

start_board =             np.array(([ 0, 0, 0, 0, 0, 0],
                                    [ 0, 0, 0, 0, 0, 0],
                                    [ 0, 0,-1,-1, 0, 0],
                                    [ 0,-1,-1, 1, 0, 0],
                                    [ 0, 1,-1, 1, 1, 0],
                                    [ 0, 1, 1, 1, 1, 1]))
def get_empty_neighbours(board,position):
    empty_neighbours = []
    n = 6
    for i in range(-1,2):
        for j in range(-1,2):
            if not(i==position[0] and j ==position[1]):
                if position[0]+i>=0 and position[0]+i<n and position[1]+j>=0 and position[1]+j<n:
                    if board[position[0]+i][position[1]+j] == 0:
                        empty_neighbours.append(((position[0]+i),(position[1]+j)))

    return empty_neighbours


start = perf_counter()
player = 1
empty_neighbours = []
for element in np.argwhere(start_board==1):
    print(element)
    empty_neighbours += get_empty_neighbours(start_board,element)
unique_empty_neighbours = set(empty_neighbours)
print(unique_empty_neighbours)
print(len(unique_empty_neighbours))
print(perf_counter()-start)





        


    
