from debug_printer import * # TODO

# ------------------------------------------------------------------------------
# Project 1
# Written by Aria Adibi, Student id: 40139168
# For COMP 6721 Section F – Fall 2019
# ------------------------------------------------------------------------------

# X= [-73.583, -73.575, -73.576]
# Y= [45.495, 45.515, 45.528]
'''
IMPORTANT:  Ambiguity in the project description.
            My Assumption: all the numbers equal to the percentile (i.e. threshold) are considered "safe."
            Note: any other posible case neither increases or decreases the code difficulty.

            The "grid residual" is considered at the top and right.
'''
import geopandas as gpd
import os
import traceback

import numpy as np

import queue
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#Global variables for accuracy and memory/speed management----------------------
COUNT_D_TYPE= np.uint64
ROUND_N_DIGITS= 8
EPS= 10 ** (-ROUND_N_DIGITS)
ROUND_N_SHOW_DIGIT= 4
TIME_LIMIT= 10 #in seconds

#Global variables given by the problem instance
INVESTIGATED_RECT= { 'TL' : (-73.59, 45.53), 'TR' : (-73.55, 45.53), 'BL' : (-73.59, 45.49), 'BR' : (-73.55, 45.49) }
MIN_X= INVESTIGATED_RECT['BL'][0];  MAX_X= INVESTIGATED_RECT['TR'][0]
MIN_Y= INVESTIGATED_RECT['BL'][1];  MAX_Y= INVESTIGATED_RECT['TR'][1]
LEN_X= round( INVESTIGATED_RECT['BR'][0] - INVESTIGATED_RECT['BL'][0], ROUND_N_DIGITS )
LEN_Y= round( INVESTIGATED_RECT['TL'][1] - INVESTIGATED_RECT['BL'][1], ROUND_N_DIGITS )

WEIGHTS= { 'SAFE_GRID' : 1, 'SAFE_DIAG' : 1.5, 'NOTSAFE_GRID' : 1.3 }

#Global variables for optimal path search algorithms
MOVEMENTS= [ (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1) ]

#Global variables for showing the figure
MAX_N_X_LABELS= 5
MAX_N_Y_LABELS= 5

#Global variables given in/inferred by the input
n= None; m= None
grid_size= None
percentile= None

#Simple index and coordinate checks --------------------------------------------
def is_in_the_grid( p ):
    #n is the number of rows of "cell" ==> number of rows of "points" = n+1, also 0 base
    if( p[0] <= -1 or p[0] >= n+1  or p[1] <= -1 or p[1] >= m+1 ):
        return False
    return True

def is_in_the_investigated_rect( p ):
    if( p[0] < (MIN_X - EPS) or p[0] > (MAX_X + EPS)  or p[1] < (MIN_Y - EPS) or p[1] > (MAX_Y + EPS) ):
        return False
    return True

def indx_to_coord( indx ):
    assert is_in_the_grid( indx ), 'Hello from indx_to_coord: Something is Wrong; Given point is not in the grid.'

    return (    round( INVESTIGATED_RECT['BL'][0] + indx[1] * grid_size, ROUND_N_DIGITS ),
                round( INVESTIGATED_RECT['BL'][1] + (n - indx[0]) * grid_size, ROUND_N_DIGITS )  )

def coord_to_indx( coord ):
    assert is_in_the_investigated_rect( coord ), 'Hello from coord_to_indx: Something is Wrong; Given coordinate is not in the investigated rectangle.'

    org_x, org_y = INVESTIGATED_RECT['BL']
    indx0= n - ( (coord[1]-org_y + EPS)  // grid_size )
    indx1= (coord[0]-org_x + EPS) // grid_size

    return ( int(indx0), int(indx1) )

#Obtaining the information -----------------------------------------------------
def get_crime_count():
    crime_count= np.zeros( (n, m), dtype= COUNT_D_TYPE )

    dir_name= os.path.dirname( os.path.abspath(__file__) )
    abs_path= os.path.dirname(dir_name) + '/Shape/crime_dt.shp'
    # os.path.join( os.path.dirname(dir_name), '/Shape/crime_dt.shp') #in first argument we go one dir back #TODO figure out
    crimes = gpd.read_file( abs_path )
    for crime in crimes.geometry:
        indx= coord_to_indx( (crime.x, crime.y) ) #returns bottom left corner
        crime_count[ indx[0] - 1 ][ indx[1] ] += 1

    return crime_count

#Finding the shortest path -----------------------------------------------------
def get_neighbors_with_move_cost( p, movements ):
    '''
    inputs:
        p: a tuple (x, y) representing an intersection point.

    outputs:
        list of tuple ( (x, y), w ) where (x, y) is a valid neighbor of p and w is the move cost from p to (x, y).
    '''
    assert is_in_the_grid(p), 'Hello from get_neighbors_with_move_cost: Something is Wrong; Given point is not in the grid' #TODO assertion?

    neighbors= []
    for move in movements:
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

def a_star_tree_search( start, goal, heuristic, time_limit ):
    '''
    The tree search implementation of A* algorithm. For the output to be complete and optimal
    the provided heuristic function must be admissible.

    inputs:
        star: start point (x, y)
        goal: goal(s) point(s) (x, y)
        heuristic: a heuristic function
        time_limit: time limit

    output:
        optimal_path:
            None, if no guaranteed optimal path is found within the time limit seconds
            Otherwise, a path, List of points (x, y). Index 0 = goal
        out_of_time:
            True if no guaranteed optimal path is found within the time limit seconds and the algorithm is still running, False otherwise.
    '''
    is_out_of_time= False
    timeout= time.time() + TIME_LIMIT
    has_optimal_path_found= False

    frontier= queue.PriorityQueue()
    frontier.put( (0 + heuristic(start, goal), start) )

    came_from= {}
    came_from[start]= None

    cost_so_far= {}
    cost_so_far[start]= 0

    while not frontier.empty():
        if( time.time() > timeout ): #checking for timeout
            is_out_of_time= True
            break

        current= frontier.get()[1]

        if is_goal(current, goal):
            has_optimal_path_found= True
            break

        for next, move_cost in get_neighbors_with_move_cost(current, MOVEMENTS):
            new_cost= cost_so_far[current] + move_cost

            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next]= new_cost
                priority= new_cost + heuristic(next, goal)
                frontier.put( (priority, next) )
                came_from[next]= current

    if( has_optimal_path_found is not True ): #no guaranteed optimal path to goal is found
        return is_out_of_time, None

    optimal_path= [goal]
    while( came_from[ optimal_path[-1] ] is not None ):
        optimal_path.append( came_from[ optimal_path[-1] ] )

        if( optimal_path[-1] not in came_from ):
            raise AssertionError('Hello from a_star_tree_search: Something is Wrong; came_from is not continuous.')
    return is_out_of_time, optimal_path

#Heuristic functions -----------------------------------------------------------
def naive_heuristic( p, goal ):
    x1, y1= p; x2, y2= goal
    n= abs(x1 - x2);    m= abs(y1 - y2)
    return min(n, m) * WEIGHTS['SAFE_DIAG'] + (max(n, m) - min(n, m)) * WEIGHTS['SAFE_GRID'] #no round needed

h= None
is_h_defined= False #TODO think of better implementation
def moving_towards_heuristic( p, goal ):
    global h #TODO understand
    global is_h_defined #TODO understand
    if( is_h_defined == False ):
        is_h_defined= True
        #initialization
        h= np.full( (n+1, m+1), np.inf, dtype= COUNT_D_TYPE )
        h[ goal[0] ][ goal[1] ]= 0

        i_steps= [-1, 1];   j_steps= [-1, 1]    #merging four fors
        for i_step in i_steps:
            for j_step in j_steps:
                #find valid movements for this direction
                movements= []
                i_valid_dirs= [0, -i_step];   j_valid_dirs= [0, -j_step]
                for i_valid_dir in i_valid_dirs:
                    for j_valid_dir in j_valid_dirs:
                        if( i_valid_dir == 0 and j_valid_dir == 0 ):
                            continue
                        movements.append( (i_valid_dir, j_valid_dir) )

                #now traverse the grid accordingly and calculate mov_towards_h
                i= goal[0]
                while( i >= 0 and i <= n ):
                    j= goal[1]
                    while( j >= 0 and j <= m ):
                        has_no_neighbor= True
                        for next, move_cost in get_neighbors_with_move_cost( (i, j), movements):
                            has_no_neighbor= False
                            h[i][j]= min( h[i][j], h[ next[0] ][ next[1] ] + move_cost )

                        if has_no_neighbor == True: #the heuristic is lenient toward blocked points with these move_cost
                            for move in movements:
                                move_cost= np.inf
                                if(move[0] == 0 or move[1] == 0):
                                    move_cost= WEIGHTS['NOTSAFE_GRID'] #any other route costs more #TODO maybe think a bit and do a bit better
                                else:
                                    move_cost= WEIGHTS['SAFE_DIAG'] #TODO again maybe do better

                                next= (i + move[0], j + move[1])    #note that these moves might contain the boundary edges
                                                                    #but due to min it is ok to be lenient about them in here
                                if( next[0] >= 0 and next[0] <= n and next[1] >= 0 and next[1] >= m ):
                                    if( (i_step < 0 and next[0] < goal[0] - i_step) or (i_step > 0 and next[0] > goal[0] - i_step) ):
                                        if( (j_step < 0 and next[1] < goal[1] - j_step) or (j_step > 0 and next[1] > goal[1] - j_step) ):
                                            h[i][j]= min( h[i][j], h[ next[0] ][ next[1] ] + move_cost )

                        j+= j_step

                    i+= i_step

    return h[ p[0] ][ p[1] ]

#Figure manipulation / showing functions ---------------------------------------
def get_format_func( labels ):

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

def show_the_thresholded_map_with_the_optimal_path(optimal_path):

    fig= plt.figure( ' ' )

    ax = fig.subplots(n, m)
    fig.subplots_adjust(hspace= 0, wspace= 0)

    for i in range(n):
        for j in range(m):
            ax[i][j].xaxis.set_major_locator( ticker.NullLocator() )
            ax[i][j].yaxis.set_major_locator( ticker.NullLocator() )
            if( no_go_zone[i][j] == True ):
                ax[i][j].set_facecolor( 'yellow' )
            else:
                ax[i][j].set_facecolor( 'blue' )

    main_ax= fig.add_subplot(1, 1, 1, facecolor= 'white', title= 'Thresholded Map with The Optimal Path')
    main_ax.patch.set_alpha(0)

    main_ax.set_xlim(MIN_X, MAX_X)
    main_ax.set_ylim(MIN_Y, MAX_Y)
    main_ax.set_xlabel('Longitude')
    main_ax.set_ylabel('Latitude')

    major_x_ticks= np.linspace(MIN_X, MAX_X, MAX_N_X_LABELS if m + 1 >= MAX_N_X_LABELS else m + 1, endpoint= True)
    main_ax.set_xticks(major_x_ticks) #only there to give some numeric perspective
    # minor_x_ticks= np.linspace(MIN_X, MAX_X, m + 1)
    # main_ax.set_xticks(minor_x_ticks, minor= True) # shows the seperating lines of cell   s
    main_ax.xaxis.set_major_formatter( ticker.FuncFormatter( get_format_func( major_x_ticks ) ) )

    major_y_ticks= np.linspace(MIN_Y, MAX_Y, MAX_N_Y_LABELS if n + 1 >= MAX_N_Y_LABELS else n + 1, endpoint= True)
    main_ax.set_yticks(major_y_ticks) #only there to give some numeric perspective
    # minor_y_ticks= np.linspace(MIN_Y, MAX_Y, n + 1)
    # main_ax.set_yticks(minor_y_ticks, minor= True) # shows the seperating lines of cells
    main_ax.yaxis.set_major_formatter( ticker.FuncFormatter( get_format_func( major_y_ticks ) ) )

    # main_ax.grid(True, which= 'minor')

    #Obtaining the optimal path
    if optimal_path is not None:
        x_optimal_path= []; y_optimal_path= []
        for p in optimal_path:
            x, y= indx_to_coord(p)
            x_optimal_path.append(x)
            y_optimal_path.append(y)

        main_ax.plot(x_optimal_path, y_optimal_path, color= 'red', linewidth= 3, label= 'An Optimal Path')

    main_ax.legend()
    plt.show()

#The Execution
if (__name__ == '__main__'):
    grid_size= float( input( 'Please enter the grid size: ' ) )
    percentile= float( input( 'Please enter the Threshold: ' ) )

    n= int( np.ceil(LEN_Y/grid_size) )
    m= int( np.ceil(LEN_X/grid_size) )
    crime_count= get_crime_count()

    print( '\nCrime Statistics:' )
    print( '\tTotal number: {}'.format( np.sum(crime_count) ) )
    print( '\tMean: {}'.format( round( np.mean(crime_count), ROUND_N_SHOW_DIGIT ) ) )
    print( '\tStandard Deviation: {}'.format( round( np.std(crime_count), ROUND_N_SHOW_DIGIT ) ) )

    #IMPORTANT: Ambiguity in the project description.
    #           My Assumption: all the numbers equal to the percentile (i.e. threshold) are considered "safe"
    no_go_zone= crime_count > np.percentile(crime_count, percentile, interpolation= 'lower')

    start= (0, 0);  goal= (n, m)
    print('''\nDo you want to specify the \'start\' and \'goal\' intersections? [y, n]
    If yes, please enter two space separated numbers representing the coordinates. Defualt is from the top left corner to the bottom right corner.''' )

    correct_input_format= False
    while not correct_input_format:
        altere= input()
        if( altere == 'y' ):
            x, y= input('Please enter \'start\' coordinate:').split(' ')
            x= float(x);    y= float(y)
            try:
                start= coord_to_indx( (x, y) )
            except AssertionError as err:
                print( err )
                print('Impossible coordinate. Please try again. Do you want to specify? [y, n]')
                continue
            except Exception as err:
                print( err )
                print( traceback.format_exc() )
                exit(1)

            x, y= input('Please enter \'goal\' coordinate:').split(' ')
            x= float(x);    y= float(y)
            try:
                goal= coord_to_indx( (x, y) )
            except AssertionError as err:
                print( err )
                print('Impossible coordinate. Please try again. Do you want to specify? [y, n]')
                continue
            except Exception as err:
                print( err )
                print( traceback.format_exc() )
                exit(1)
            correct_input_format= True

        elif( altere == 'n' ):
            correct_input_format= True

        else:
            print('Wrong answer format. Please write \'y\' or \'n\'.')

    # is_out_of_time, optimal_path= a_star_tree_search(start, goal, naive_heuristic, TIME_LIMIT)
    is_out_of_time, optimal_path= a_star_tree_search(start, goal, moving_towards_heuristic, TIME_LIMIT)

    if is_out_of_time == True:
        print('Time is up. A (guaranteed) optimal path is not found.')
    elif optimal_path == None:
        print('Due to blocks, no path is found. Please change the map and try again.')
    else:
        print('The optimal path is found and will be shown shortly. DONE.')

    show_the_thresholded_map_with_the_optimal_path(optimal_path)
