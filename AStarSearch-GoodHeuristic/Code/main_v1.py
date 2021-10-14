from debug_printer import * # TODO

# ------------------------------------------------------------------------------
# Project 1
# Written by Aria Adibi, Student id: 40139168
# For COMP 6721 Section (your lab section) – Fall 2019 #TODO
# ------------------------------------------------------------------------------

'''
IMPORTANT:  Ambiguity in the project description.
            My Assumption: all the numbers equal to the percentile (i.e. threshold) are considered "safe."
            Note: any other posible case neither increases or decreases the code difficulty.
'''

import numpy as np
import queue
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#Global variables for accuracy and memory/speed management----------------------
COUNT_D_TYPE= np.uint32
EPS= 10 ** (-8)

#Global variables given by the problem instance
INVESTIGATED_RECT= { 'TL' : (-73.59, 45.53), 'TR' : (-73.55, 45.53), 'BL' : (-73.59, 45.49), 'BR' : (-73.55, 45.49) }
MIN_X= INVESTIGATED_RECT['BL'][0];  MAX_X= INVESTIGATED_RECT['TR'][0]
MIN_Y= INVESTIGATED_RECT['BL'][1];  MAX_Y= INVESTIGATED_RECT['TR'][1]

WEIGHTS= { 'SAFE_GRID' : 1, 'SAFE_DIAG' : 1.5, 'NOTSAFE_GRID' : 1.3 }

#Global variables for optimal path search algorithms
MOVEMENTS= [ (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1) ]

#Global variables for showing the figure
MAX_N_X_LABELS= 5
MAX_N_Y_LABELS= 5
X_LABELS= np.linspace(MIN_X, MAX_X, MAX_N_X_LABELS, endpoint= True)
Y_LABELS= np.linspace(MIN_Y, MAX_Y, MAX_N_Y_LABELS, endpoint= True)

#TODO
GRID_SIZE= 0.01 #TODO get from input
PERCENTILE= None #TODO get from input
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#get the imput #TODO for debugging
#-------------------------------------------------------------------------------
n= int( input('Enter #rows: ') )
m= int( input('Enter #columns: ') )

# crime_count= np.zeros( (n, m), dtype= COUNT_D_TYPE )

# print( 'Please enter the matrix:' )
# for i in range(n):
#     inps= input().split(' ')
#     for j in range(m):
#         crime_count[i][j]= COUNT_D_TYPE( inps[j] )

# debug_print( 'crime_count', globals(), locals() ) #TODO
no_go_zone= np.zeros( (n, m), dtype= np.bool )

print( 'Please enter the matrix:' )
for i in range(n):
    inps= input().split(' ')
    for j in range(m):
        no_go_zone[i][j]= np.bool( ord( inps[j] ) - ord( '0' ) )

debug_print('no_go_zone', globals(), locals()) #TODO
#-------------------------------------------------------------------------------

#IMPORTANT: Ambiguity in the project description.
#           My Assumption: all the numbers equal to the percentile (i.e. threshold) are considered "safe"
# no_go_zone= crime_count > np.percentile(crime_count, PERCENTILE, interpolation= 'lower')
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#Simple index and coordinate checks --------------------------------------------
def is_in_the_grid( p ):
    #n is the number of rows of "cell" ==> number of rows of "points" = n+1, also 0 base
    if( p[0] <= -1 or p[0] >= n+1  or p[1] <= -1 or p[1] >= m+1 ):
        return False
    return True

def is_in_the_investigated_rect( p ):
    if( p[0] < MIN_X or p[0] > MAX_X  or p[1] < MIN_Y or p[1] > MAX_Y ):
        return False
    return True

def indx_to_coord( indx ):
    assert is_in_the_grid( indx ), 'Hello from indx_to_coord: Something is Wrong; Given point is not in the grid.'

    return (    INVESTIGATED_RECT['BL'][0] + indx[1] * GRID_SIZE,
                INVESTIGATED_RECT['BL'][1] + (n - indx[0]) * GRID_SIZE  )

def coord_to_indx( coord ):
    assert is_in_the_investigated_rect(p), 'Hello from coord_to_indx: Something is Wrong; Given coordinate is not in the investigated rectangle.'

    org_x, org_y = INVESTIGATED_RECT['BL']
    ind0= n - ( (coord[1]-org_y + EPS)  // GRID_SIZE )
    idx1= (coord[0]-org_x + EPS) // GRID_SIZE

    return (ind0, ind1)

#Finding the shortest path -----------------------------------------------------
def get_neighbors_with_move_cost( p ):
    '''
    inputs:
        p: a tuple (x, y) representing an intersection point.

    outputs:
        list of tuple ( (x, y), w ) where (x, y) is a valid neighbor of p and w is the move cost from p to (x, y).
    '''
    assert is_in_the_grid(p), 'Hello from get_neighbors_with_move_cost: Something is Wrong; Given point is not in the grid' #TODO assertion?

    neighbors= []
    for move in MOVEMENTS:
        neighbor= (p[0] + move[0], p[1] + move[1])

        if( not is_in_the_grid(neighbor) ):
            continue
        #excluding the boundary edges
        elif(   (p[0] == 0 and neighbor[0] == 0) or (p[0] == n and neighbor[0] == n) or
                (p[1] == 0 and neighbor[1] == 0) or (p[1] == m and neighbor[1] == m)  ):
            continue
        else:
            #Since we excluded the boundary edges, all horizontal and vertical moves have 2 corresponding cells
            the_cells= [    [ min(p[0], neighbor[0]), min(p[1], neighbor[1]) ], [np.inf, np.inf]  ] #in diag one (only [0]), in horizontal and vertical two
            if( move[0] == 0 ):     #horizontal move
                the_cells[1]= ( the_cells[0][0] - 1, the_cells[0][1] )
            elif( move[1] == 0 ):   #vertical move
                the_cells[1]= ( the_cells[0][0], the_cells[0][1] - 1)
            else:                   #diagonal move
                pass

            w= np.inf #the associated weight if crossing is possible
            if( the_cells[1] == [np.inf, np.inf] ): #diagonal moves
                if( no_go_zone[ the_cells[0][0] ][ the_cells[0][1] ] ): #not possible to cross; due to danger
                    continue
                else:
                    w= WEIGHTS['SAFE_DIAG']
            else:
                is_not_safe_cell_0= no_go_zone[ the_cells[0][0] ][ the_cells[0][1] ]
                is_not_safe_cell_1= no_go_zone[ the_cells[1][0] ][ the_cells[1][1] ]

                if( is_not_safe_cell_0 and is_not_safe_cell_1 ): #not possible to use the edge; due to danger
                    continue
                elif( is_not_safe_cell_0 or is_not_safe_cell_1 ): #possible to cross; on the boundary of danger
                    w= WEIGHTS['NOTSAFE_GRID']
                else: #possible to cross; perfectly safe
                    w= WEIGHTS['SAFE_GRID']

        neighbors.append( (neighbor, w) )
    return neighbors

def is_goal( p, goal ):
    return p == goal

def a_star_tree_search( start, goal, heuristic ):
    '''
    The tree search implementation of A* algorithm. For the output to be complete and optimal
    the provided heuristic function must be admissible.

    inputs:
        star: start point (x, y)
        goal: goal(s) point(s) (x, y)
        heuristic: a heuristic function

    output:
        optimal_path:
            None, if no path is found
            List of points (x, y) in the path. Index 0 = goal
    '''
    frontier= queue.PriorityQueue()
    frontier.put( (0 + heuristic(start, goal), start) )

    came_from= {}
    came_from[start]= None

    cost_so_far= {}
    cost_so_far[start]= 0

    while not frontier.empty():
        current= frontier.get()[1]

        if is_goal(current, goal):
            break

        for next, move_cost in get_neighbors_with_move_cost(current):
            new_cost= cost_so_far[current] + move_cost

            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next]= new_cost
                priority= new_cost + heuristic(next, goal)
                frontier.put( (priority, next) )
                came_from[next]= current

    if( goal not in came_from ): #no path to goal is found
        return None

    optimal_path= [goal]
    while( came_from[ optimal_path[-1] ] is not None ):
        optimal_path.append( came_from[ optimal_path[-1] ] )

        if( optimal_path[-1] not in came_from ):
            raise AssertionError('Hello from a_star_tree_search: Something is Wrong; came_from is not continuous.')
    return optimal_path

#Heuristic functions -----------------------------------------------------------
def naive_heuristic( a, b ):
    x1, y1 = a; x2, y2 = b
    n= abs(x1 - x2);    m= abs(y1 - y2)
    return min(n, m) * WEIGHTS['SAFE_DIAG'] + (max(n, m) - min(n, m)) * WEIGHTS['SAFE_GRID']

#Figure manipulation / showing functions ---------------------------------------
def get_format_func( axis ):
    if axis == 'x':
        labels= X_LABELS
    elif axis == 'y':
        labels= Y_LABELS
    else:
        raise AssertionError('Hello from get_format_func: Something is Wrong; axis is neither \'x\' or \'y\'.')

    def format_func( value, tick_number ):
        found= False
        for label in labels:
            if( label - EPS <= value <= label + EPS ):
                found= True

        if found:
            return '{:0.2f}'.format(value)
        else:
            return

    return format_func

def figure_v1():
    fig= plt.figure(facecolor= 'white')
    ax= fig.add_subplot(1, 1, 1, facecolor= 'white', title= 'Thresholded Map with The Optimal Path') #TODO color

    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(MIN_Y, MAX_Y)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.set_xticks(X_LABELS)
    minor_x_ticks= np.linspace(MIN_X, MAX_X, (MAX_X - MIN_X + EPS) // GRID_SIZE + 1)
    ax.set_xticks(minor_x_ticks, minor= True)
    ax.xaxis.set_major_formatter( ticker.FuncFormatter( get_format_func('x') ) )

    ax.set_yticks(Y_LABELS)
    minor_y_ticks= np.linspace(MIN_Y, MAX_Y, (MAX_Y - MIN_Y + EPS) // GRID_SIZE + 1)
    ax.set_yticks(minor_y_ticks, minor= True)
    ax.yaxis.set_major_formatter( ticker.FuncFormatter( get_format_func('y') ) )

    #TODO ----------------------------------------------------------------------
    # right_inset_ax = fig.add_axes([0, 0, GRID_SIZE / (MAX_X - MIN_X), GRID_SIZE / (MAX_Y - MIN_Y)], facecolor= 'yellow')
    # # right_inset_ax = fig.add_axes([0.2, 0.2, 0.4, 0.4], facecolor= 'green')
    # right_inset_ax.xaxis.set_major_locator( ticker.NullLocator() )
    # right_inset_ax.yaxis.set_major_locator( ticker.NullLocator() )
    #
    # right_inset_ax= ax.add_child_axes([0, 0, GRID_SIZE / (MAX_X - MIN_X), GRID_SIZE / (MAX_Y - MIN_Y)], facecolor= 'green')
    # ax.add
    #---------------------------------------------------------------------------

    ax.grid(True, which= 'both')

    # n= 4;   m=4
    # ax2= fig.add_subplot(n, m, 1, facecolor= 'blue')
    # ax2.xaxis.set_major_locator( ticker.NullLocator() )
    # ax2.yaxis.set_major_locator( ticker.NullLocator() )
    # ax3= fig.add_subplot(n, m, 2, facecolor= 'yellow')
    # ax3.xaxis.set_major_locator( ticker.NullLocator() )
    # ax3.yaxis.set_major_locator( ticker.NullLocator() )

    #Obtaining the optimal path
    if optimal_path is not None:
        x_optimal_path= []; y_optimal_path= []
        for p in optimal_path:
            x, y= indx_to_coord(p)
            x_optimal_path.append(x)
            y_optimal_path.append(y)

        ax.plot(x_optimal_path, y_optimal_path, color= 'red', linewidth= 3, label= 'An Optimal Path')

    ax.legend()
    plt.show()

def figure_v2(optimal_path):



    fig= plt.figure( figsize= (n, m) )
    main_ax= fig.add_subplot(1, 1, 1, facecolor= 'white', title= 'Thresholded Map with The Optimal Path') #TODO color
    ax = fig.subplots(n, m)
    fig.subplots_adjust(hspace=0, wspace=0)

    for i in range(n):
        for j in range(m):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].set_facecolor( 'yellow' )

    # main_ax.set_xlim(MIN_X, MAX_X)
    # main_ax.set_ylim(MIN_Y, MAX_Y)
    # main_ax.set_xlabel('Longitude')
    # main_ax.set_ylabel('Latitude')

    # main_ax.set_xticks(X_LABELS)
    # minor_x_ticks= np.linspace(MIN_X, MAX_X, (MAX_X - MIN_X + EPS) // GRID_SIZE + 1)
    # main_ax.set_xticks(minor_x_ticks, minor= True)
    # main_ax.xaxis.set_major_formatter( ticker.FuncFormatter( get_format_func('x') ) )
    #
    # main_ax.set_yticks(Y_LABELS)
    # minor_y_ticks= np.linspace(MIN_Y, MAX_Y, (MAX_Y - MIN_Y + EPS) // GRID_SIZE + 1)
    # main_ax.set_yticks(minor_y_ticks, minor= True)
    # main_ax.yaxis.set_major_formatter( ticker.FuncFormatter( get_format_func('y') ) )

    # main_ax.grid(True, which= 'both')

    #Obtaining the optimal path
    if optimal_path is not None:
        # debug_print('optimal_path', globals(), locals()) #TODO
        x_optimal_path= []; y_optimal_path= []
        for p in optimal_path:
            # debug_print('p', globals(), locals()) #TODO
            x, y= indx_to_coord(p)
            # debug_print('here', globals(), locals(), is_str= 1) #TODO
            x_optimal_path.append(x)
            y_optimal_path.append(y)

        main_ax.plot(x_optimal_path, y_optimal_path, color= 'red', linewidth= 3, label= 'An Optimal Path')

    main_ax.legend()
    plt.show()

#The Execution
if (__name__ == '__main__'):
    # get_format_func('w')
    start= (0, 0); goal= (n, m)
    optimal_path= a_star_tree_search(start, goal, naive_heuristic)

    if optimal_path == None:
        print('Due to blocks, no path is found. Please change the map and try again.')
    else:
        print('The optimal path is found and shown. DONE.')

    # figure_v1()
    figure_v2(optimal_path)
