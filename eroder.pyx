# distutils: language=c++
# cython: language_level=3, cdivision=True

from libcpp.vector cimport vector
from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as cnp

cdef double distance(double a, double b):
    return sqrt(a*a + b*b)


cdef double get_water_velocity_vector(int x, int y, int size_x, int size_y, double[:, :, :] flow):
    cdef double left = flow[x - 1][y][0] - flow[x][y][2] if x > 1 else 0.0 
    cdef double right = flow[x][y][0] - flow[x + 1][y][2] if x < size_x - 1 else 0.0 
    cdef double bottom = flow[x][y - 1][1] - flow[x][y][3] if y > 1 else 0.0 
    cdef double top = flow[x][y][1] - flow[x][y + 1][3] if y < size_y - 1 else 0.0 
    
    #return sqrt(0.5 * (sqr(left + right) + sqr(top + bottom)))
    return distance(0.5 * (left + right), 0.5 * (top + bottom))


# I decided to code it in the same way normals are computed
# so the average steepness, or, the first derivative
cdef double get_steepness(int x, int y, int size_x, int size_y, double[:, :] heightmap):
    cdef int[4] delta_x = [ 1, 0, -1, 0 ]
    cdef int[4] delta_y = [ 0, 1, 0, -1 ]
    cdef int i = 0
    cdef double delta_h
        
    # x, y, magnitude
    cdef double[3] steepness = [0.0, 0.0, 0.0]
    
    for i in range(4):        
        if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
            delta_h = heightmap[x][y] - heightmap[x + delta_x[i]][y + delta_y[i]]
            steepness[0] += delta_x[i] * delta_h
            steepness[1] += delta_y[i] * delta_h
    #return sqrt(sqr(steepness[0]) + sqr(steepness[1]))
    return distance(steepness[0], steepness[1])


# similar to a second derivative
# positive = bump
# negative = hole
cdef double get_convexness(int x, int y, int size_x, int size_y, double[:, :] heightmap):
    cdef int[4] delta_x = [ 1, 0, -1, 0 ]
    cdef int[4] delta_y = [ 0, 1, 0, -1 ]
    cdef int i = 0
    cdef double delta_h
        
    # x, y, magnitude
    cdef double[3] convexness = [0.0, 0.0, 0.0]
    
    for i in range(4):        
        if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
            delta_h = heightmap[x][y] - heightmap[x + delta_x[i]][y + delta_y[i]]
            convexness[0] += fabs(delta_x[i]) * delta_h
            convexness[1] += fabs(delta_y[i]) * delta_h
    #return sqrt(sqr(convexness[0]) + sqr(convexness[1]))
    return distance(convexness[0], convexness[1])


cpdef erode(unsigned int size_x, unsigned int size_y, list _heightmap, list _water, list _previous_water, list _sediment, list _flow):
    # our universal iterators
    cdef unsigned int i, j, x, y;

    # this is edited manually, I just use it to make sure that Blender loads the correct file (often doesn't)
    print("eroder version = 18")

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

    cdef unsigned int step = 0, steps = 1
    cdef int[4][3] neighbors = [[1, 0, 0], [0, 1, 1], [-1, 0, 2], [0, -1, 3]]
    cdef double length = 1.0
    cdef double area = length * length
    cdef double g = 9.81
    cdef double c = 0.1
    cdef double delta_t = 0.1 
    cdef int[4] delta_x = [ 1, 0, -1, 0 ]
    cdef int[4] delta_y = [ 0, 1, 0, -1 ]

    cdef vector[vector[double]] water2
    cdef double[4] acceleration = [0.0, 0.0, 0.0, 0.0]
    cdef double delta_height
    cdef double scaling
    cdef double flow_velocity
    cdef double local_shape_factor
    cdef double sediment_capacity
    cdef double out_volume_sum

    print("starting erosion...") 
        
    for step in range(steps):
        print("step no. " + str(step))

        # reset water2
        water2.resize(size_x)
        for x in range(size_x):
            water2.at(x).resize(size_y, 0.0)   

        for x in range(size_x):
            for y in range(size_y):
                if water[x][y] > 0:
                    ######################################################
                    # MOVE WATER
                    ######################################################
                    
                    height_here = heightmap[x][y] + water[x][y]
                    acceleration = [0.0, 0.0, 0.0, 0.0]
                    out_volume_sum = 0.0
                    
                    # calculate the height differences with neighbors and flow
                    for i in range(4):
                        if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                            height_neighbor = heightmap[x + delta_x[i]][y + delta_y[i]] + water[x + delta_x[i]][ y + delta_y[i]]
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
                        if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                            flow[x][y][i] *= scaling
                            water2[x + delta_x[i]][y + delta_y[i]] += flow[x][y][i] * delta_t / area
                            
                    water2[x][y] += water[x][y] - (out_volume_sum / area)
                    

                    ######################################################
                    # FORCE EROSION
                    ######################################################
                    average_height = 0.5 * (water[x][y] + previous_water[x][y])

                    # this max function doesn't really have a physical basis, but it makes sure that small amounts of water don't erode ridiculous amounts
                    average_height = max(average_height, 0.2)

                    flow_velocity = get_water_velocity_vector(x, y, size_x, size_y, flow)

                    # steepness and convexness
                    local_shape_factor = 1.0 * (get_steepness(x, y, size_x, size_y, heightmap) + 0.25 * get_convexness(x, y, size_x, size_y, heightmap))
                    sediment_capacity = (flow_velocity / (length * average_height)) * c * local_shape_factor
                    
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
                        
                else:
                    # might be useless?
                    water2[x][y] += water[x][y]
                
    
        # update terrain and water heights
        for x in range(size_x):
            for y in range(size_y):
                previous_water[x][y] = water[x][y]
                water[x][y] = water2[x][y]

    print("done")

    # return a ctuple (or tuple? tbh nobody cares, it works) of Python lists by converting a memoryview into a numpy array and then converting that into a regular array
    return (np.array(heightmap).tolist(), np.array(water).tolist(), np.array(previous_water).tolist(), np.array(sediment).tolist(), np.array(flow).tolist()) 