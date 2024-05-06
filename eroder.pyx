# distutils: language=c++
# cython: language_level=3, cdivision=True

from libcpp.vector cimport vector
from libc.math cimport sqrt, fabs


cdef double get_water_velocity_vector(int x, int y, int size_x, int size_y, vector[vector[vector[double]]] flow) noexcept:
    cdef double left = flow[x - 1][y][0] - flow[x][y][2] if x > 1 else 0.0 
    cdef double right = flow[x][y][0] - flow[x + 1][y][2] if x < size_x - 1 else 0.0 
    cdef double bottom = flow[x][y - 1][1] - flow[x][y][3] if y > 1 else 0.0 
    cdef double top = flow[x][y][1] - flow[x][y + 1][3] if y < size_y - 1 else 0.0 
    
    return sqrt(0.5 * ((left + right)**2 + (top + bottom)**2))


# I decided to code it in the same way normals are computed
# so the average steepness, or, the first derivative
cdef double get_steepness(int x, int y, int size_x, int size_y, vector[vector[double]] heightmap) noexcept:
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
    return sqrt(steepness[0]**2 + steepness[1]**2)


# similar to a second derivative
# positive = bump
# negative = hole
cdef double get_convexness(int x, int y, int size_x, int size_y, vector[vector[double]] heightmap) noexcept:
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
    return sqrt(convexness[0]**2 + convexness[1]**2)


cpdef erode(unsigned int size_x, unsigned int size_y, vector[vector[double]] heightmap, vector[vector[double]] water, vector[vector[double]] previous_water, vector[vector[double]] sediment, vector[vector[vector[double]]] flow):
    # our universal iterators
    cdef unsigned int i, j, x, y;

    print("eroder version = 9")

    """     
    # convert from lame Python types to glorious Cython types
    cdef vector[vector[double]] heightmap
    heightmap.resize(size_x)
    for i in range(size_x):
        heightmap[i].resize(size_y)
        for j in range(size_y):
            heightmap[i][j] = _heightmap[i][j] + 1 
    """

    cdef unsigned int step = 0, steps = 10
    cdef int[4][3] neighbors = [[1, 0, 0], [0, 1, 1], [-1, 0, 2], [0, -1, 3]]
    cdef double length = 1.0
    cdef double reciprocal_length = 1.0 / length
    cdef double area = length * length
    cdef double reciprocal_area = 1.0 / area
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
                    flow_sum = 0.0
                    
                    # calculate the height differences with neighbors and flow
                    for i in range(4):
                        if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                            height_neighbor = heightmap[x + delta_x[i]][y + delta_y[i]] + water[x + delta_x[i]][ y + delta_y[i]]
                            delta_height = height_here - height_neighbor
                            acceleration[i] = g * delta_height * reciprocal_length
                            flow[x][y][i] = max(0.0, flow[x][y][i] + (delta_t * acceleration[i] * length * length))
                            flow_sum += delta_t * flow[x][y][i]
                            
                    # scale flow
                    scaling = 1.0
                    if flow_sum > length * length * water[x][y] and flow_sum > 0:
                        scaling = length * length * water[x][y] / flow_sum
                        flow_sum = length * length * water[x][y]
                        
                    # add to neighbors
                    for i in range(4):
                        if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                            flow[x][y][i] *= scaling
                            water2[x + delta_x[i]][y + delta_y[i]] += flow[x][y][i] * delta_t * reciprocal_area
                            
                    water2[x][y] += water[x][y] - (flow_sum * reciprocal_area)
                    

                    ######################################################
                    # FORCE EROSION
                    ######################################################
                    average_height = 0.5 * (water[x][y] + previous_water[x][y])

                    # this max function doesn't really have a physical basis, but it makes sure that small amounts of water don't erode ridiculous amounts
                    average_height = max(average_height, 0.2)

                    flow_velocity = get_water_velocity_vector(x, y, size_x, size_y, flow)

                    # steepness and convexness
                    local_shape_factor = 1.0 * (get_steepness(x, y, size_x, size_y, heightmap) + 0.25 * get_convexness(x, y, size_x, size_y, heightmap))
                    sediment_capacity = (flow_velocity * reciprocal_length / average_height) * c * local_shape_factor
                    
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
                    water2[x][y] = water[x][y]
                            
                previous_water[x][y] = water[x][y]
                
    
        # update terrain and water heights
        #water = water2
        for x in range(size_x):
            for y in range(size_y):
                water[x][y] = water2[x][y]

    print("done")

    # If possible, C values and C++ objects are automatically
    # converted to Python objects at need.
    return (heightmap, water, previous_water, sediment, flow)  # so here, the vector will be copied into a Python list.