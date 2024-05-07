# distutils: language=c++
# cython: language_level=3, cdivision=True

from libcpp.vector cimport vector
from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand

cdef extern from "stdlib.h":
    int RAND_MAX
cdef double BETTER_RAND_MAX = RAND_MAX

cdef double distance(double a, double b):
    return sqrt(a*a + b*b)


cdef double random(double low=0.0, double high=1.0):
    return low + (rand() / BETTER_RAND_MAX) * (high - low)


cdef double get_water_velocity_vector(int x, int y, int size_x, int size_y, double[:, :, :] flow):
    cdef double left = flow[(x - 1) % size_x][y][0] - flow[x][y][2] # if x > 1 else 0.0 
    cdef double right = flow[x][y][0] - flow[(x + 1) % size_x][y][2] # if x < size_x - 1 else 0.0 
    cdef double bottom = flow[x][(y - 1) % size_y][1] - flow[x][y][3] # if y > 1 else 0.0 
    cdef double top = flow[x][y][1] - flow[x][(y + 1) % size_y][3] # if y < size_y - 1 else 0.0 
    
    #return sqrt(0.5 * (sqr(left + right) + sqr(top + bottom)))
    return distance(0.5 * (left + right), 0.5 * (top + bottom))


# I decided to code it in the same way normals are computed
# so the average steepness, or, the first derivative
# nvm, it's sine now
cdef double get_steepness(int x, int y, int size_x, int size_y, double[:, :] heightmap):
    cdef double dx = abs(heightmap[(x - 1) % size_x][y] - heightmap[(x + 1) % size_x][y])
    cdef double dy = abs(heightmap[x][(y - 1) % size_y] - heightmap[x][(y + 1) % size_y])
  
    # assumes length = 1.0
    return sqrt(dx**2 + dy**2) / sqrt(dx**2 + dy**2 + 1.0**2)


# similar to a second derivative
# positive = bump
# negative = hole
cdef double get_convexness(int x, int y, int size_x, int size_y, double[:, :] heightmap):
    cdef int[4] delta_x = [ 1, 0, -1, 0 ]
    cdef int[4] delta_y = [ 0, 1, 0, -1 ]
    cdef int i = 0
    cdef double delta_h
        
    # x, y, magnitude
    cdef double cx = 0.0, cy = 0.0
    
    for i in range(4):        
        #if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
        delta_h = heightmap[x][y] - heightmap[(x + delta_x[i]) % size_x][(y + delta_y[i]) % size_y]
        cx += 0.5 * fabs(delta_x[i]) * delta_h
        cy += 0.5 * fabs(delta_y[i]) * delta_h
    
    # assumes length = 1.0
    return 0.5 * (cx + cy) / sqrt((0.5 * (cx + cy))**2 + 1.0**2)


# adding more water than the max_depth will not speed up the erosion
# source: that Balazs paper
cdef double max_erosion_depth(double depth):
    cdef double max_depth = 0.45
    if depth >= max_depth:
        return 1.0
    elif 0 < depth < max_depth:
        return 1.0 - (max_depth - depth) / max_depth
    else:
        return 0.0


cpdef erode(unsigned int size_x, unsigned int size_y, list _heightmap, list _water, list _previous_water, list _sediment, list _flow):
    # our universal iterators
    cdef unsigned int i, j, x, y
    cdef int si, sj

    # this is edited manually, I just use it to make sure that Blender loads the correct file (often doesn't)
    print("eroder version = 21")

    # converts lists into memory views
    cdef cnp.ndarray[cnp.float64_t, ndim=2] __heightmap = np.array(_heightmap, dtype=np.float64)
    cdef double[:, :] heightmap = __heightmap
    cdef cnp.ndarray[cnp.float64_t, ndim=2] __water = np.array(_water, dtype=np.float64)
    cdef double[:, :] water = __water
    cdef cnp.ndarray[cnp.float64_t, ndim=2] __previous_water = np.array(_previous_water, dtype=np.float64)
    cdef double[:, :] previous_water = __previous_water
    cdef cnp.ndarray[cnp.float64_t, ndim=2] __sediment = np.array(_sediment, dtype=np.float64)
    cdef double[:, :] sediment = __sediment
    cdef cnp.ndarray[cnp.float64_t, ndim=3] __flow = np.array(_flow, dtype=np.float64)
    cdef double[:, :, :] flow = __flow

    #cdef cnp.ndarray[cnp.float64_t, ndim=3] soil_names = np.array(_soil_names, dtype=np.float64)
    #cdef cnp.ndarray[cnp.float64_t, ndim=3] soil_thickness = np.array(_soil_thickness, dtype=np.float64)


    cdef unsigned int step = 0, steps = 5
    cdef int[4][3] neighbors = [[1, 0, 0], [0, 1, 1], [-1, 0, 2], [0, -1, 3]]
    cdef double length = 1.0
    cdef double area = length * length
    cdef double g = 9.81
    cdef double c = 0.25
    cdef double delta_t = 0.1 
    cdef int[4] delta_x = [ 1, 0, -1, 0 ]
    cdef int[4] delta_y = [ 0, 1, 0, -1 ]

    cdef vector[vector[double]] water2
    cdef vector[vector[double]] sediment2
    cdef vector[vector[double]] heightmap2
    cdef double[4] acceleration = [0.0, 0.0, 0.0, 0.0]
    cdef double delta_height
    cdef double scaling
    cdef double flow_velocity
    cdef double local_shape_factor
    cdef double sediment_capacity
    cdef double out_volume_sum
    cdef double[3][3] neighbors_delta_height
    cdef double talus_angle_tan = 0.577  # 30 degrees

    print("starting erosion...") 

    # resize vectors
    water2.resize(size_x)
    sediment2.resize(size_x)
    heightmap2.resize(size_x)
    for x in range(size_x):
        water2.at(x).resize(size_y, 0.0)
        sediment2.at(x).resize(size_y, 0.0)
        heightmap2.at(x).resize(size_y, 0.0)
        

    for step in range(steps):
        print("step no. " + str(step))

        # reset vectors
        for x in range(size_x):
            for y in range(size_y):
                water2[x][y] = 0.0 
                sediment2[x][y] = 0.0 
                heightmap2[x][y] = heightmap[x][y]

        
        ######################################################
        # THERMAL EROSION
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                neighbors_delta_height = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] 
                delta_h_sum = 0.0
                max_delta = 0.0

                for si in range(-1, 2):
                    for sj in range(-1, 2):
                        if si != 0 or sj != 0:
                            delta = heightmap[x][y] - heightmap[(x + si) % size_x][(y + sj) % size_y]

                            if delta > max_delta:
                                max_delta = delta

                            if delta / length >= talus_angle_tan:
                                neighbors_delta_height[si + 1][sj + 1] = delta
                                delta_h_sum += delta
                                
                
                height_to_move = delta_t * 0.5 * max_delta * random(0.9, 1.1)  # maybe times local hardness

                if delta_h_sum > 0:
                    for si in range(-1, 2):
                        for sj in range(-1, 2):
                            if si != 0 or sj != 0:
                                heightmap2[(x + si) % size_x][(y + sj) % size_y] += height_to_move * neighbors_delta_height[si + 1][sj + 1] / delta_h_sum
                    
                heightmap2[x][y] -= height_to_move

        for x in range(size_x):
            for y in range(size_y):
                heightmap[x][y] = heightmap2[x][y]      

        ######################################################
        # MOVE WATER
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                if water[x][y] > 0:                    
                    height_here = heightmap[x][y] + water[x][y]
                    acceleration = [0.0, 0.0, 0.0, 0.0]
                    out_volume_sum = 0.0
                    
                    # calculate the height differences with neighbors and flow
                    for i in range(4):
                        #if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                        height_neighbor = heightmap[(x + delta_x[i]) % size_x][(y + delta_y[i]) % size_y] + water[(x + delta_x[i]) % size_x][(y + delta_y[i]) % size_y]
                        delta_height = height_here - height_neighbor
                        acceleration[i] = g * delta_height / length
                        flow[x][y][i] = max(0.0, flow[x][y][i] + (delta_t * acceleration[i] * length * length))
                        out_volume_sum += delta_t * flow[x][y][i]
                            
                    # scale flow
                    scaling = 1.0
                    column_water = length * length * water[x][y]
                    
                    if out_volume_sum > column_water and out_volume_sum > 0:
                        scaling = column_water / out_volume_sum
                        out_volume_sum = column_water
                        
                    # add to neighbors
                    for i in range(4):
                        #if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                        flow[x][y][i] *= scaling
                        water2[(x + delta_x[i]) % size_x][(y + delta_y[i]) % size_y] += flow[x][y][i] * delta_t / area

                        # move sediment
                        sediment_fraction = flow[x][y][i] * delta_t / column_water
                        sediment2[(x + delta_x[i]) % size_x][(y + delta_y[i]) % size_y] += sediment[x][y] * sediment_fraction
                        
                    water2[x][y] += water[x][y] - (out_volume_sum / area)
                    sediment2[x][y] += sediment[x][y] * (1 - out_volume_sum/column_water)


        for x in range(size_x):
            for y in range(size_y):
                previous_water[x][y] = water[x][y]
                water[x][y] = water2[x][y]
                sediment[x][y] = sediment2[x][y]
                sediment2[x][y] = 0.0


        ######################################################
        # FORCE EROSION
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                if water[x][y] > 0:
                    average_height = 0.5 * (water[x][y] + previous_water[x][y])

                    # this max function doesn't really have a physical basis, but it makes sure that small amounts of water don't erode ridiculous amounts
                    average_height = max(average_height, 0.1)

                    flow_velocity = get_water_velocity_vector(x, y, size_x, size_y, flow) / (length * average_height)

                    # steepness and convexness
                    #local_shape_factor = max(0.0, 1.0 * max(0.05, get_steepness(x, y, size_x, size_y, heightmap)) + 0.0 * get_convexness(x, y, size_x, size_y, heightmap))
                    sediment_capacity = flow_velocity * c * max(0.05, get_steepness(x, y, size_x, size_y, heightmap)) * max_erosion_depth(water[x][y])
                    
                    if sediment[x][y] <= sediment_capacity:
                        # erode
                        amount = 0.5 * (sediment_capacity - sediment[x][y])
                        heightmap[x][y] -= amount
                        sediment[x][y] += amount
                    else:
                        # deposit
                        amount = 0.25 * (sediment[x][y] - sediment_capacity)
                        heightmap[x][y] += amount
                        sediment[x][y] -= amount
                        
                """ else:
                    # might be useless?
                    water2[x][y] += water[x][y]
                    sediment2[x][y] += sediment[x][y] """
                

        ######################################################
        # DIFFUSE SEDIMENT
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                # using a filter where the neighbors have weights of 0.5 and the column itself has a weight of 1.0
                # so the sum is 1.0 + 4*0.5 = 3.0
                base = 1.0 * water[x][y]
                for i in range(4):
                    base += 0.5 * water[(x + delta_x[i]) % size_x][(y + delta_y[i]) % size_y]

                if base > 0:
                    sediment2[x][y] += sediment[x][y] * 1.0 * water[x][y] / base
                    for i in range(4):
                        sediment2[(x + delta_x[i]) % size_x][(y + delta_y[i]) % size_y] += sediment[x][y] * 0.5 * water[(x + delta_x[i]) % size_x][(y + delta_y[i]) % size_y] / base
                else:
                    sediment2[x][y] += sediment[x][y]

        # update sediment again
        for x in range(size_x):
            for y in range(size_y):
                sediment[x][y] = sediment2[x][y]



    print("done")

    # return a ctuple (or tuple? tbh nobody cares, it works) of Python lists by converting a memoryview into a numpy array and then converting that into a regular array
    return (np.array(heightmap).tolist(), np.array(water).tolist(), np.array(previous_water).tolist(), np.array(sediment).tolist(), np.array(flow).tolist()) 