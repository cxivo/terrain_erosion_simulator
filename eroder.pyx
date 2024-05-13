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


# this is needed because Cython sometimes thinks that -1 % 10 == -1 
cdef unsigned int mod(int a, int b):
    return (a + b) % b


# poor man's equality
cdef bint eq(double a, double b):
    if a - 0.00000001 < b < a + 0.00000001:
        return True
    return False


cdef double get_water_velocity_vector(int x, int y, int size_x, int size_y, double[:, :, :] flow):
    cdef double left = flow[mod(x - 1, size_x)][y][0] - flow[x][y][2] # if x > 1 else 0.0 
    cdef double right = flow[x][y][0] - flow[mod(x + 1, size_x)][y][2] # if x < size_x - 1 else 0.0 
    cdef double bottom = flow[x][mod(y - 1, size_y)][1] - flow[x][y][3] # if y > 1 else 0.0 
    cdef double top = flow[x][y][1] - flow[x][mod(y + 1, size_y)][3] # if y < size_y - 1 else 0.0 
    
    #return sqrt(0.5 * (sqr(left + right) + sqr(top + bottom)))
    return distance(0.5 * (left + right), 0.5 * (top + bottom))


# I decided to code it in the same way normals are computed
# so the average steepness, or, the first derivative
# nvm, it's sine now
cdef double get_steepness(int x, int y, int size_x, int size_y, double[:, :] heightmap):
    cdef double dx = abs(heightmap[mod(x - 1, size_x)][y] - heightmap[mod(x + 1, size_x)][y])
    cdef double dy = abs(heightmap[x][mod(y - 1, size_y)] - heightmap[x][mod(y + 1, size_y)])
  
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
        delta_h = heightmap[x][y] - heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)]
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



##################################################
# SOILS AND ROCKS

cdef struct Soil:
    double organic  # height of organic matter in soil, meters
    # 0 - sand
    # 1 - gravel
    # 2 - rocks
    # 3 - boulders
    double[4] rocks  # heights of various coarsenesses of materials, meters


cdef struct Bedrock:
    double height
    int type


# "terrain horizon" is the name of the fact that various soils and rocks are ordered in layers
cdef struct Horizon:
    Soil topsoil
    Soil subsoil
    vector[Bedrock] bedrock


##################################################
# GEOLOGY CONSTANTS

cdef int ORGANIC = 4  # just the id in arrays
cdef int TYPE = 0
cdef int HEIGHT = 1

cdef double MAX_TOPSOIL_HEIGHT = 1.5

cdef Soil EMPTY_SOIL
EMPTY_SOIL.organic = 0.0
for i in range(4):
    EMPTY_SOIL.rocks[i] = 0.0



##################################################
# HELPER FUNCTIONS

cdef Soil new_soil(double organic, double rock0, double rock1, double rock2, double rock3):
    cdef Soil soil
    soil.organic = organic
    soil.rocks[0] = rock0
    soil.rocks[1] = rock1
    soil.rocks[2] = rock2
    soil.rocks[3] = rock3
    return soil


cdef Soil new_soil_rock(double rock, int index):
    cdef Soil soil = EMPTY_SOIL
    soil.rocks[index] = rock
    return soil


cdef double get_soil_height(Soil& soil):
    cdef double h = soil.organic
    cdef unsigned int i

    for i in range(4):
        h += soil.rocks[i]

    return h


# returns Soil with same ratios, but with cummulative height of 1.0
cdef Soil normalized(Soil soil):
    cdef double soil_height = get_soil_height(soil)

    if soil_height == 0.0:
        return EMPTY_SOIL

    soil.organic /= soil_height
    for i in range(4):
        soil.rocks[i] /= soil_height
    
    return soil


# returns Soil with same ratios, but scaled by fraction
cdef Soil soil_fraction(Soil soil, double fraction):
    if not (0.0 <= fraction <= 1.0):
        print("Invalid fraction value: " + str(fraction))

    # sometimes 1.0000000000000002 comes up and I don't like it
    if fraction > 1.0:
        fraction = 1.0

    soil.organic *= fraction
    for i in range(4):
        soil.rocks[i] *= fraction
    
    return soil


cdef Soil add(Soil a, Soil b):
    a.organic += b.organic
    for i in range(4):
        a.rocks[i] += b.rocks[i]
    
    return a


# MODIFIES ARGUMENT
cdef void add_to(Soil& a, Soil b):
    a.organic += b.organic
    for i in range(4):
        a.rocks[i] += b.rocks[i]


# makes sure that after needed_height of material is added, the topsoil doesn't go over the max height
# transfers part of topsoil into subsoil
# MODIFIES ARGUMENT
cdef void organize_horizon(Horizon& horizon, double needed_height):
    cdef double topsoil_height = get_soil_height(horizon.topsoil)
    cdef double fraction_remove
    cdef double amount

    if topsoil_height + needed_height > MAX_TOPSOIL_HEIGHT and topsoil_height > 0:
        # fraction of topsoil which needs to get transferred to subsoil
        fraction_remove = max(min((topsoil_height + needed_height - MAX_TOPSOIL_HEIGHT) / topsoil_height, 1.0), 0.0)

        amount = horizon.topsoil.organic * fraction_remove
        horizon.subsoil.organic += amount
        horizon.topsoil.organic -= amount

        for i in range(4):
            amount = horizon.topsoil.rocks[i] * fraction_remove
            horizon.subsoil.rocks[i] += amount
            horizon.topsoil.rocks[i] -= amount


cdef double get_soil_talus(Soil soil):
    soil = normalized(soil)

    if get_soil_height(soil) <= 0.0:
        return float("inf")

    cdef double talus = 1.4 * soil.organic
    talus += 0.7 * soil.rocks[0]
    talus += 0.8 * soil.rocks[1]
    talus += 1.0 * soil.rocks[2]
    talus += 1.3 * soil.rocks[3]

    return talus

##################################################
# EROSION AND DEPOSITION


# MODIFIES ARGUMENT
cdef double remove_soil(Horizon& horizon, double amount):
    cdef double topsoil_height = get_soil_height(horizon.topsoil)
    cdef double fraction_remove = 1.0
    cdef double removed = 0.0

    # topsoil
    if topsoil_height > 0:
        fraction_remove = min(amount / topsoil_height, 1.0)
        horizon.topsoil = soil_fraction(horizon.topsoil, 1.0 - fraction_remove)
        removed += fraction_remove * topsoil_height

    # and also subsoil
    cdef double subsoil_height = get_soil_height(horizon.subsoil)

    # topsoil
    if amount > topsoil_height and subsoil_height > 0.0:
        fraction_remove = min((amount - topsoil_height) / subsoil_height, 1.0)
        horizon.subsoil = soil_fraction(horizon.subsoil, 1.0 - fraction_remove)
        removed += fraction_remove * subsoil_height
    
    if not eq(amount, removed):
        print("discrepancy! amount=" + str(amount) + ", removed=" + str(removed))

    return removed


# MODIFIES ARGUMENT
cdef void deposit_soil(Horizon& destination, Soil& soil):
    cdef unsigned int i

    # prepare place in topsoil for new stuff
    organize_horizon(destination, get_soil_height(soil))

    destination.topsoil.organic += soil.organic
    for i in range(4):
        destination.topsoil.rocks[i] += soil.rocks[i]

    # make sure topsoil is not too thick (might delete later)
    organize_horizon(destination, 0.0)


# MODIFIES ARGUMENT
cdef double erode_organic_amount(Horizon& horizon, double requested_amount):
    cdef double amount_remove
    requested_amount = max(0.0, requested_amount)

    if get_soil_height(horizon.topsoil) > 0.0:
        amount_remove = min(horizon.topsoil.organic, requested_amount)
        horizon.topsoil.organic -= amount_remove
        return amount_remove
    elif get_soil_height(horizon.subsoil) > 0.0:
        amount_remove = min(horizon.subsoil.organic, requested_amount)
        horizon.subsoil.organic -= amount_remove
        return amount_remove
    else:
        return 0.0


# MODIFIES ARGUMENT
cdef double erode_rocks_amount(Horizon& horizon, double requested_amount, int index):
    cdef double amount_remove
    requested_amount = max(0.0, requested_amount)

    if get_soil_height(horizon.topsoil) > 0.0:
        amount_remove = min(horizon.topsoil.rocks[index], requested_amount)
        horizon.topsoil.rocks[index] -= amount_remove
        return amount_remove
    elif get_soil_height(horizon.subsoil) > 0.0:
        amount_remove = min(horizon.subsoil.rocks[index], requested_amount)
        horizon.subsoil.rocks[index] -= amount_remove
        return amount_remove
    else:
        return 0.0



##################################################
##################################################
cpdef erode(unsigned int size_x, unsigned int size_y, list _heightmap, list _water, list _previous_water, list _sediment, list _flow, list _regolith, list _topsoil, list _subsoil, list _bedrock):
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

    """ cdef cnp.ndarray[cnp.float64_t, ndim=3] __sediment = np.array(_sediment, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=3] __topsoil = np.array(_topsoil, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=3] __subsoil = np.array(_subsoil, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=4] __bedrock = np.array(_bedrock, dtype=np.float64) """
    #cdef double[:, :, :] flow = __flow

    cdef cnp.ndarray[cnp.float64_t, ndim=3] __flow = np.array(_flow, dtype=np.float64)
    cdef double[:, :, :] flow = __flow
    cdef cnp.ndarray[cnp.float64_t, ndim=2] __regolith = np.array(_regolith, dtype=np.float64)
    cdef double[:, :] regolith = __regolith

    #cdef cnp.ndarray[cnp.float64_t, ndim=3] soil_names = np.array(_soil_names, dtype=np.float64)
    #cdef cnp.ndarray[cnp.float64_t, ndim=3] soil_thickness = np.array(_soil_thickness, dtype=np.float64)


    cdef unsigned int step = 0, steps = 1
    cdef int[4][3] neighbors = [[1, 0, 0], [0, 1, 1], [-1, 0, 2], [0, -1, 3]]
    cdef double length = 1.0
    cdef double area = length * length
    cdef double g = 9.81
    cdef double erosion_constant = 0.1
    cdef double max_penetration_depth = 0.01
    cdef double regolith_constant = 0.5
    cdef double delta_t = 0.1 
    cdef int[4] delta_x = [ 1, 0, -1, 0 ]
    cdef int[4] delta_y = [ 0, 1, 0, -1 ]

    cdef double organic_constant = 0.12
    cdef double[4][4] rock_matrix_linear = [[0.15,  0.0,  0.0,  0.0],
                                            [0.04,  0.08, 0.0,  0.0],
                                            [0.04,  0.02, 0.04, 0.0],
                                            [0.04,  0.02, 0.01, 0.02]]
    cdef double[4][4] rock_matrix_constant = [[ 0.0,   0.0,   0.0,  0.0],
                                              [-0.01, -0.1,   0.0,  0.0],
                                              [-0.02, -0.2,  -0.3,  0.0],
                                              [-0.03, -0.25, -0.4, -0.5]]

    cdef vector[vector[double]] water2
    cdef vector[vector[Soil]] sediment
    cdef vector[vector[Soil]] sediment2
    cdef vector[vector[Horizon]] terrain
    cdef vector[vector[vector[vector[Soil]]]] soil_flow
    cdef vector[vector[double]] regolith2
    cdef vector[vector[double]] heightmap2
    cdef double[4] acceleration = [0.0, 0.0, 0.0, 0.0]
    cdef double delta_height
    cdef double scaling
    cdef double flow_velocity
    cdef double local_shape_factor
    cdef double sediment_capacity
    cdef double out_volume_sum
    cdef double[3][3] neighbors_delta_height
    cdef Soil soil

    print("starting erosion...") 

    # initialize vectors
    water2.resize(size_x)
    sediment.resize(size_x)
    sediment2.resize(size_x)
    terrain.resize(size_x)
    soil_flow.resize(size_x)
    heightmap2.resize(size_x)
    regolith2.resize(size_x)
    for x in range(size_x):
        water2.at(x).resize(size_y, 0.0)
        sediment.at(x).resize(size_y)
        sediment2.at(x).resize(size_y)
        soil_flow.at(x).resize(size_y)
        terrain.at(x).resize(size_y)
        heightmap2.at(x).resize(size_y, 0.0)
        regolith2.at(x).resize(size_y, 0.0)
        for y in range(size_y):
            # problems may arise, who knows
            if water[x][y] < 0.0:
                water[x][y] = 0.0


            sediment[x][y].organic = _sediment[y + size_y * x][ORGANIC]
            for i in range(4):
                sediment[x][y].rocks[i] = _sediment[y + size_y * x][i]

            soil_flow[x][y].resize(3)
            for i in range(3):
                soil_flow[x][y][i].resize(3)

            

            # init terrain
            terrain[x][y].topsoil.organic = _topsoil[y + size_y * x][ORGANIC]
            for i in range(4):
                terrain[x][y].topsoil.rocks[i] = _topsoil[y + size_y * x][i]

            terrain[x][y].subsoil.organic = _subsoil[y + size_y * x][ORGANIC]
            for i in range(4):
                terrain[x][y].subsoil.rocks[i] = _subsoil[y + size_y * x][i]

            vector_size = len(_bedrock[y + size_y * x]) // 2
            terrain[x][y].bedrock.resize(vector_size)
            for i in range(vector_size):
                terrain[x][y].bedrock[i].type = round(_bedrock[y + size_y * x][2*i])
                terrain[x][y].bedrock[i].height = _bedrock[y + size_y * x][2*i + 1]


    for step in range(steps):
        print("step no. " + str(step))
   
        ######################################################
        # GRAVITATIONAL EROSION
        ######################################################

        for x in range(size_x):
            for y in range(size_y):
                # clear flows
                heightmap2[x][y] = 0.0
                for si in range(-1, 2):
                    for sj in range(-1, 2):
                        soil_flow[x][y][si + 1][sj + 1] = EMPTY_SOIL


                # which neighbors will get a piece of our terrain
                multiplier = 1.0
                for soil in [terrain[x][y].topsoil, terrain[x][y].subsoil]:
                    
                    delta_h_sum = 0.0
                    max_delta = 0.0

                    if get_soil_height(soil) > 0.0:
                        neighbors_delta_height = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] 
                        for si in range(-1, 2):
                            for sj in range(-1, 2):
                                if si != 0 or sj != 0:
                                    delta = heightmap[x][y] - heightmap[mod(x + si, size_x)][mod(y + sj, size_y)]

                                    # if should erode
                                    if delta / (length * sqrt(si**2 + sj**2)) >= get_soil_talus(soil) * multiplier:
                                        # largest difference
                                        if delta > max_delta:
                                            max_delta = delta

                                        # save for later reference
                                        neighbors_delta_height[si + 1][sj + 1] = delta
                                        delta_h_sum += delta
                                        
                        # 0.5 is not needed, just somthing 1 >= x > 0
                        to_distribute = 0.5 * min(0.5 * max_delta, get_soil_height(soil))
    
                        height_fraction = to_distribute / get_soil_height(soil)

                        # how much will flow OUT of this cell later
                        # misusing this array
                        heightmap2[x][y] = to_distribute

                        # assign flows
                        if delta_h_sum > 0:
                            for si in range(-1, 2):
                                for sj in range(-1, 2):
                                    if si != 0 or sj != 0:
                                        fraction = height_fraction * neighbors_delta_height[si + 1][sj + 1] / delta_h_sum
                                        soil_flow[x][y][si + 1][sj + 1] = soil_fraction(soil, fraction)
                        
                        # erode only the topsoil
                        break
                    else:
                        # if the topsoil is empty, try the subsoil
                        # make lower layer more tough
                        multiplier *= 1.25

        # assign flows
        for x in range(size_x):
            for y in range(size_y):
                # remove soil from this column
                remove_soil(terrain[x][y], heightmap2[x][y])
                heightmap[x][y] -= heightmap2[x][y]

                # deposit soil from neighbors
                for si in range(-1, 2):
                    for sj in range(-1, 2):
                        if si != 0 or sj != 0:
                            deposit_soil(terrain[x][y], soil_flow[mod(x + si, size_x)][mod(y + sj, size_y)][1 - si][1 - sj])
                            heightmap[x][y] += get_soil_height(soil_flow[mod(x + si, size_x)][mod(y + sj, size_y)][1 - si][1 - sj])

        ######################################################
        # MOVE WATER
        ######################################################

        for x in range(size_x):
            for y in range(size_y):
                sediment2[x][y] = EMPTY_SOIL
                regolith2[x][y] = 0.0
                water2[x][y] = 0.0

        for x in range(size_x):
            for y in range(size_y):
                if water[x][y] > 0:                    
                    height_here = heightmap[x][y] + water[x][y]
                    acceleration = [0.0, 0.0, 0.0, 0.0]
                    out_volume_sum = 0.0
                    
                    # calculate the height differences with neighbors and flow
                    for i in range(4):
                        #if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                        height_neighbor = heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] + water[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)]
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
                        water2[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] += flow[x][y][i] * delta_t / area

                        # move sediment
                        sediment_fraction = flow[x][y][i] * delta_t / column_water
                        add_to(sediment2[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)], soil_fraction(sediment[x][y], sediment_fraction))
                        regolith2[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] += regolith[x][y] * sediment_fraction
                        
                    # what remains here
                    water2[x][y] += water[x][y] - (out_volume_sum / area)
                    add_to(sediment2[x][y], soil_fraction(sediment[x][y], (1 - out_volume_sum/column_water)))
                    regolith2[x][y] += regolith[x][y] * (1 - out_volume_sum/column_water)

        # assign
        for x in range(size_x):
            for y in range(size_y):
                previous_water[x][y] = water[x][y]
                water[x][y] = water2[x][y]
                sediment[x][y] = sediment2[x][y]
                regolith[x][y] = regolith2[x][y]


        ######################################################
        # FORCE EROSION
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                if water[x][y] > 0:
                    average_height = 0.5 * (water[x][y] + previous_water[x][y])

                    # this max function doesn't really have a physical basis, but it makes sure that small amounts of water don't erode ridiculous amounts
                    average_height = max(average_height, 0.05)

                    flow_velocity = get_water_velocity_vector(x, y, size_x, size_y, flow) / (length * average_height)

                    # steepness and convexness
                    #local_shape_factor = max(0.0, 1.0 * max(0.05, get_steepness(x, y, size_x, size_y, heightmap)) + 0.0 * get_convexness(x, y, size_x, size_y, heightmap))
                    # no erosion_constant
                    water_info = 0.25 * flow_velocity * max(0.05, get_steepness(x, y, size_x, size_y, heightmap)) * max_erosion_depth(water[x][y])
                    
                    # organic
                    sediment_capacity = organic_constant * water_info
                    if sediment[x][y].organic <= sediment_capacity:
                        # erode
                        requested_amount = 0.5 * (sediment_capacity - sediment[x][y].organic)
                        amount = erode_organic_amount(terrain[x][y], requested_amount)
                        heightmap[x][y] -= amount
                        sediment[x][y].organic += amount
                        #water[x][y] += amount
                    else:
                        # deposit
                        amount = 0.25 * (sediment[x][y].organic - sediment_capacity)
                        deposit_soil(terrain[x][y], new_soil(amount, 0, 0, 0, 0))
                        heightmap[x][y] += amount
                        sediment[x][y].organic -= amount
                        #water[x][y] -= amount

                    # erode rocks
                    for original in range(4):
                        for sedimented in range(4):
                            sediment_capacity = max(rock_matrix_linear[original][sedimented] * water_info + rock_matrix_constant[original][sedimented], 0.0)
                            if sediment[x][y].rocks[sedimented] <= sediment_capacity:
                            # erode
                                requested_amount = 0.5 * (sediment_capacity - sediment[x][y].rocks[original])
                                amount = erode_rocks_amount(terrain[x][y], requested_amount, original)
                                heightmap[x][y] -= amount
                                sediment[x][y].rocks[sedimented] += amount
                                #water[x][y] += amount
                            
                    # deposit rocks
                    for rock in range(4):      
                        sediment_capacity = max(rock_matrix_linear[rock][rock] * water_info + rock_matrix_constant[rock][rock], 0.0)
                        if sediment[x][y].rocks[rock] > sediment_capacity:
                            # deposit
                            amount = 0.25 * (sediment[x][y].rocks[rock] - sediment_capacity)
                            deposit_soil(terrain[x][y], new_soil_rock(amount, rock))
                            heightmap[x][y] += amount
                            sediment[x][y].rocks[rock] -= amount
                            #water[x][y] -= amount
                        

        ######################################################
        # REGOLITH EROSION
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                if water[x][y] > 0:
                    # max regolith
                    max_regolith_thickness = max_penetration_depth
                    if water[x][y] < max_penetration_depth:
                        max_regolith_thickness = water[x][y]

                    # always maintain max regolith (no idea why)
                    if regolith[x][y] < max_regolith_thickness:
                        # regolithise (not a real word)
                        amount = erode_rocks_amount(terrain[x][y], max_regolith_thickness - regolith[x][y], 0)
                        heightmap[x][y] -= amount
                        regolith[x][y] += amount
                    else:
                        # deposit
                        amount = regolith[x][y] - max_regolith_thickness
                        deposit_soil(terrain[x][y], new_soil_rock(amount, 0))
                        heightmap[x][y] += amount
                        regolith[x][y] -= amount

                
        for x in range(size_x):
            for y in range(size_y):
                water2[x][y] = water[x][y]
                sediment2[x][y] = EMPTY_SOIL

        ######################################################
        # DIFFUSE SEDIMENT
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                # using a filter where the neighbors have weights of 0.5 and the column itself has a weight of 1.0
                # so the sum is 1.0 + 4*0.5 = 3.0
                base = 1.0 * water[x][y]
                for i in range(4):
                    base += 0.5 * water[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)]

                if base > 0:
                    add_to(sediment2[x][y], soil_fraction(sediment[x][y], 1.0 * water[x][y] / base))
                    #water2[x][y] -= sediment[x][y] * (1.0 - (1.0 * water[x][y] / base))
                    for i in range(4):
                        fraction = 0.5 * water[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] / base
                        add_to(sediment2[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)], soil_fraction(sediment[x][y], fraction))
                        #water2[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] += amount
                else:
                    add_to(sediment2[x][y], sediment[x][y])

        # update sediment again
        for x in range(size_x):
            for y in range(size_y):
                sediment[x][y] = sediment2[x][y]
                water[x][y] = water2[x][y]


    print("done")

    #####################################
    # POSTPROCESSING (misusing the name, I know)

    # visually lower water which is zero in height
    cdef int neighbors_count = 0
    cdef double height_sum = 0.0
    for x in range(size_x):
        for y in range(size_y):
            if water[x][y] <= 0.0:
                # calculate average height of neighbors
                neighbors_count = 0
                height_sum = 0.0
                for si in range(-1, 2):
                    for sj in range(-1, 2):
                        if water[mod(x+si, size_x)][mod(y+sj, size_y)] > 0.0:
                            neighbors_count += 1
                            height_sum += water[mod(x+si, size_x)][mod(y+sj, size_y)] + heightmap[mod(x+si, size_x)][mod(y+sj, size_y)]
                
                # set water height to average of the neighbors, if below 0
                if neighbors_count > 0:
                    water[x][y] = min(0.0, height_sum / neighbors_count - heightmap[x][y])
                
                # if all neighbors are dry or too high
                if neighbors_count == 0 or water[x][y] == 0.0:
                    water[x][y] = -1.0  # just put it deep down


    # create pixel data for images
    cdef cnp.ndarray[cnp.float64_t, ndim=1] _topsoil_texture = np.zeros(shape=(4 * size_x * size_y))
    cdef double[:] topsoil_texture = _topsoil_texture
    # RGBA
    cdef double[4] color_topsoil = [1.0, 1.0, 0.0, 1.0]

    for x in range(size_x):
        for y in range(size_y):
            if get_soil_height(terrain[x][y].topsoil) > 0.0:
                topsoil_texture[4 * (y + size_y * x) + 1] = 1.0
                topsoil_texture[4 * (y + size_y * x) + 3] = 1.0  # Alpha channel
            if get_soil_height(terrain[x][y].subsoil) > 0.0:
                topsoil_texture[4 * (y + size_y * x) + 0] = 1.0
                topsoil_texture[4 * (y + size_y * x) + 3] = 1.0  # Alpha channel

                #for i in range(4):
                #    topsoil_texture[4 * (y + size_y * x) + i] = color_topsoil[i]


    # create Python lists
    _topsoil = [[terrain[x][y].topsoil.rocks[0], 
                  terrain[x][y].topsoil.rocks[1],
                  terrain[x][y].topsoil.rocks[2],
                  terrain[x][y].topsoil.rocks[3],
                  terrain[x][y].topsoil.organic] for y in range(size_y) for x in range(size_x)]

    _subsoil = [[terrain[x][y].subsoil.rocks[0], 
                  terrain[x][y].subsoil.rocks[1],
                  terrain[x][y].subsoil.rocks[2],
                  terrain[x][y].subsoil.rocks[3],
                  terrain[x][y].subsoil.organic] for y in range(size_y) for x in range(size_x)]

    _bedrock = [[thing for thing in [terrain[x][y].bedrock[i].type, terrain[x][y].bedrock[i].height]]
                 for i in range(len(terrain[x][y].bedrock)) for y in range(size_y) for x in range(size_x)]

    _sediment = [[sediment[x][y].rocks[0],
                   sediment[x][y].rocks[1],
                   sediment[x][y].rocks[2],
                   sediment[x][y].rocks[3],
                   sediment[x][y].organic] for y in range(size_y) for x in range(size_x)]

    # return a ctuple (or tuple? tbh nobody cares, it works) of Python lists by converting a memoryview into a numpy array and then converting that into a regular array
    return (np.array(heightmap).tolist(), np.array(water).tolist(), np.array(previous_water).tolist(), _sediment, np.array(flow).tolist(), np.array(regolith).tolist(), _topsoil, _subsoil, _bedrock, np.array(topsoil_texture).tolist()) 