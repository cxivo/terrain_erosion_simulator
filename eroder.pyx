# distutils: language=c++
# cython: language_level=3, cdivision=True, boundscheck=False

from libcpp.vector cimport vector
from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand
from cython.parallel import prange
from libc.time cimport time,time_t


cdef extern from "stdlib.h":
    int RAND_MAX
cdef double BETTER_RAND_MAX = RAND_MAX


cdef inline double random(double low=0.0, double high=1.0) nogil:
    return low + (rand() / BETTER_RAND_MAX) * (high - low)


# this is needed because Cython sometimes thinks that -1 % 10 == -1 
cdef inline unsigned int mod(int a, int b) nogil:
    return (a + b) % b


# poor man's equality
cdef inline bint eq(double a, double b) nogil:
    if a - 0.00000001 < b < a + 0.00000001:
        return True
    return False


cdef inline double water_velocity(int x, int y, int size_x, int size_y, double[:, :, :] flow) noexcept nogil:
    cdef double left = flow[mod(x - 1, size_x)][y][0] - flow[x][y][2] # if x > 1 else 0.0 
    cdef double right = flow[x][y][0] - flow[mod(x + 1, size_x)][y][2] # if x < size_x - 1 else 0.0 
    cdef double bottom = flow[x][mod(y - 1, size_y)][1] - flow[x][y][3] # if y > 1 else 0.0 
    cdef double top = flow[x][y][1] - flow[x][mod(y + 1, size_y)][3] # if y < size_y - 1 else 0.0 
    
    return sqrt((0.5 * (left + right))**2 + (0.5 * (top + bottom))**2)


cdef (double, double) water_velocity_vector(int x, int y, int size_x, int size_y, double[:, :, :] flow) noexcept nogil:
    cdef double left = flow[mod(x - 1, size_x)][y][0] - flow[x][y][2] # if x > 1 else 0.0 
    cdef double right = flow[x][y][0] - flow[mod(x + 1, size_x)][y][2] # if x < size_x - 1 else 0.0 
    cdef double bottom = flow[x][mod(y - 1, size_y)][1] - flow[x][y][3] # if y > 1 else 0.0 
    cdef double top = flow[x][y][1] - flow[x][mod(y + 1, size_y)][3] # if y < size_y - 1 else 0.0 
    
    return (0.5 * (left + right), 0.5 * (top + bottom))


# I decided to code it in the same way normals are computed
# so the average steepness, or, the first derivative
# nvm, it's sine now
cdef inline double get_steepness(int x, int y, int size_x, int size_y, double[:, :] heightmap) noexcept nogil:
    cdef double dx = abs(heightmap[mod(x - 1, size_x)][y] - heightmap[mod(x + 1, size_x)][y])
    cdef double dy = abs(heightmap[x][mod(y - 1, size_y)] - heightmap[x][mod(y + 1, size_y)])
  
    # assumes length = 1.0
    return sqrt(dx**2 + dy**2) / sqrt(dx**2 + dy**2 + 1.0**2)


# similar to a second derivative
# positive = bump
# negative = hole
cdef double get_convexness(int x, int y, int size_x, int size_y, double[:, :] heightmap) noexcept nogil:
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
cdef inline double max_erosion_depth(double depth) noexcept nogil:
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


cdef struct Water:
    double height
    double previous
    double regolith
    Soil sediment;
    double velocity_x
    double velocity_y

##################################################
# GEOLOGY CONSTANTS

cdef int ORGANIC = 4  # just the id in arrays
cdef int TYPE = 0
cdef int HEIGHT = 1
cdef double infinity = float("inf")

cdef double MAX_TOPSOIL_HEIGHT = 1.5
cdef double MAX_WATER_SPEED = 20.0  # source: made up

cdef Soil EMPTY_SOIL
EMPTY_SOIL.organic = 0.0
for i in range(4):
    EMPTY_SOIL.rocks[i] = 0.0

cdef Water EMPTY_WATER
EMPTY_WATER.height = 0.0
EMPTY_WATER.previous = 0.0
EMPTY_WATER.regolith = 0.0
EMPTY_WATER.sediment = EMPTY_SOIL
EMPTY_WATER.velocity_x = 0.0
EMPTY_WATER.velocity_y = 0.0

##################################################
# HELPER FUNCTIONS

cdef inline Soil new_soil(double organic, double rock0, double rock1, double rock2, double rock3) noexcept nogil:
    cdef Soil soil
    soil.organic = organic
    soil.rocks[0] = rock0
    soil.rocks[1] = rock1
    soil.rocks[2] = rock2
    soil.rocks[3] = rock3
    return soil


cdef inline Soil new_soil_rock(double rock, int index) noexcept nogil:
    cdef Soil soil = EMPTY_SOIL
    soil.rocks[index] = rock
    return soil


cdef inline double get_soil_height(Soil& soil) noexcept nogil:
    cdef double h = soil.organic
    cdef unsigned int i

    for i in range(4):
        h += soil.rocks[i]

    return h


# returns Soil with same ratios, but with cummulative height of 1.0
cdef inline Soil normalized(Soil soil) noexcept nogil:
    cdef double soil_height = get_soil_height(soil)

    if soil_height == 0.0:
        return EMPTY_SOIL

    soil.organic /= soil_height
    for i in range(4):
        soil.rocks[i] /= soil_height
    
    return soil


# returns Soil with same ratios, but scaled by fraction
cdef inline Soil soil_fraction(Soil soil, double fraction) noexcept nogil:
    #if not (0.0 <= fraction <= 1.0):
    #    print("Invalid fraction value: " + str(fraction))

    # sometimes 1.0000000000000002 comes up and I don't like it
    if fraction > 1.0:
        fraction = 1.0

    soil.organic *= fraction
    for i in range(4):
        soil.rocks[i] *= fraction
    
    return soil


cdef inline Soil add_soil(Soil a, Soil b) noexcept nogil:
    a.organic += b.organic
    for i in range(4):
        a.rocks[i] += b.rocks[i]
    
    return a


# MODIFIES ARGUMENT
cdef inline void add_to_soil(Soil& a, Soil b) noexcept nogil:
    a.organic += b.organic
    for i in range(4):
        a.rocks[i] += b.rocks[i]


# makes sure that after needed_height of material is added, the topsoil doesn't go over the max height
# transfers part of topsoil into subsoil
# MODIFIES ARGUMENT
cdef void organize_horizon(Horizon& horizon, double needed_height) noexcept nogil:
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


cdef double get_soil_talus(Soil soil) noexcept nogil:
    soil = normalized(soil)

    if get_soil_height(soil) <= 0.0:
        return infinity

    cdef double talus = 1.4 * soil.organic
    talus += 0.7 * soil.rocks[0]
    talus += 0.8 * soil.rocks[1]
    talus += 1.0 * soil.rocks[2]
    talus += 1.3 * soil.rocks[3]

    return talus


##################################################
# WATER

# returns Water with same ratios, but scaled by fraction
cdef inline Water water_fraction(Water water, double fraction) noexcept nogil:
    #if not (0.0 <= fraction <= 1.0):
    #    print("Invalid fraction value: " + str(fraction))

    # sometimes 1.0000000000000002 comes up and I don't like it
    if fraction > 1.0:
        fraction = 1.0

    water.height *= fraction
    # previous water is handled differently
    water.regolith *= fraction
    water.sediment = soil_fraction(water.sediment, fraction)
    # velocity remains unchanged

    return water


cdef inline Water add_water(Water a, Water b) noexcept nogil:
    cdef Water c = a
    add_to_water(c, b)
    return c


# MODIFIES ARGUMENT
cdef void add_to_water(Water& a, Water b) noexcept nogil:
    if b.height <= 0.0:
        return

    cdef double velocity_multiplier = a.height / (a.height + b.height)

    a.height += b.height
    a.regolith += b.regolith
    add_to_soil(a.sediment, b.sediment)
    # previous water is handled differently
    a.velocity_x = a.velocity_x * velocity_multiplier + b.velocity_x * (1.0 - velocity_multiplier)
    a.velocity_y = a.velocity_y * velocity_multiplier + b.velocity_y * (1.0 - velocity_multiplier)


cdef inline Water speedup(Water water, double x, double y) noexcept nogil:
    water.velocity_x += x
    water.velocity_y += y
    return water


# MODIFIES ARGUMENT
cdef inline void clip_speed(Water& water) noexcept nogil:
    cdef double scaling = 1.0
    cdef double velocity = sqrt(water.velocity_x**2 + water.velocity_y**2)

    if velocity > MAX_WATER_SPEED:
        scaling = MAX_WATER_SPEED / velocity
        water.velocity_x *= scaling
        water.velocity_y *= scaling


##################################################
# EROSION AND DEPOSITION


# MODIFIES ARGUMENT
cdef double remove_soil(Horizon& horizon, double amount) noexcept nogil:
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
    
    """ if not eq(amount, removed):
        print("discrepancy! amount=" + str(amount) + ", removed=" + str(removed)) """

    return removed


# MODIFIES ARGUMENT
cdef void deposit_soil(Horizon& destination, Soil& soil) noexcept nogil:
    cdef unsigned int i

    # prepare place in topsoil for new stuff
    organize_horizon(destination, get_soil_height(soil))

    destination.topsoil.organic += soil.organic
    for i in range(4):
        destination.topsoil.rocks[i] += soil.rocks[i]

    # make sure topsoil is not too thick (might delete later)
    organize_horizon(destination, 0.0)


# MODIFIES ARGUMENT
cdef double erode_organic_amount(Horizon& horizon, double requested_amount) noexcept nogil:
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
cdef double erode_rocks_amount(Horizon& horizon, double requested_amount, int index) noexcept nogil:
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


# MODIFIES ARGUMENT
cdef double erode_organic_amount_only_from(Soil& soil, double requested_amount) noexcept nogil:
    cdef double amount_remove
    requested_amount = max(0.0, requested_amount)

    amount_remove = min(soil.organic, requested_amount)
    soil.organic -= amount_remove
    return amount_remove


# MODIFIES ARGUMENT
cdef double erode_rocks_amount_only_from(Soil& soil, double requested_amount, int index) noexcept nogil:
    cdef double amount_remove
    requested_amount = max(0.0, requested_amount)

    amount_remove = min(soil.rocks[index], requested_amount)
    soil.rocks[index] -= amount_remove
    return amount_remove



##################################################
##################################################
cpdef erode(unsigned int steps, double delta_t, double erosion_constant, double max_penetration_depth, double regolith_constant, double inertial_erosion_constant, bint is_river, double source_height, unsigned int size_x, unsigned int size_y, list _heightmap, list _water, list _previous_water, list _sediment, list _flow, list _regolith, list _topsoil, list _subsoil, list _bedrock, list _velocity_x, list _velocity_y):
    # our universal iterators
    cdef unsigned int i, j, x, y
    cdef int si, sj

    # this is edited manually, I just use it to make sure that Blender loads the correct file (often doesn't)
    print("eroder version = 23")

    # converts lists into memory views
    cdef cnp.ndarray[cnp.float64_t, ndim=2] __heightmap = np.array(_heightmap, dtype=np.float64)
    cdef double[:, ::1] heightmap = __heightmap
    cdef cnp.ndarray[cnp.float64_t, ndim=3] __flow = np.array(_flow, dtype=np.float64)
    cdef double[:, :, :] flow = __flow


    cdef unsigned int step = 0
    cdef int[4][3] neighbors = [[1, 0, 0], [0, 1, 1], [-1, 0, 2], [0, -1, 3]]
    cdef double length = 1.0
    cdef double area = length * length
    cdef double g = 9.81
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

    cdef vector[vector[Water]] water
    cdef vector[vector[Water]] water2
    cdef vector[vector[Horizon]] terrain
    cdef vector[vector[vector[Water]]] water_flow
    cdef vector[vector[vector[vector[Soil]]]] soil_flow
    cdef vector[vector[double]] heightmap2
    """ cdef double[4] acceleration = [0.0, 0.0, 0.0, 0.0]
    cdef double delta_height
    cdef double scaling
    cdef double flow_velocity
    cdef double local_shape_factor
    cdef double sediment_capacity
    cdef double out_volume_sum
    cdef double[3][3] neighbors_delta_height
    cdef Soil soil """

    print("starting erosion...") 

    # initialize vectors
    water.resize(size_x)
    water2.resize(size_x)
    terrain.resize(size_x)
    soil_flow.resize(size_x)
    water_flow.resize(size_x)
    heightmap2.resize(size_x)

    for x in range(size_x):
        water.at(x).resize(size_y, EMPTY_WATER)
        water2.at(x).resize(size_y, EMPTY_WATER)
        soil_flow.at(x).resize(size_y)
        water_flow.at(x).resize(size_y)
        terrain.at(x).resize(size_y)
        heightmap2.at(x).resize(size_y, 0.0)
        for y in range(size_y):
            # water 
            water[x][y].height = _water[x][y]
            if water[x][y].height < 0.0:
                water[x][y].height = 0.0

            water[x][y].previous = _previous_water[x][y]

            water[x][y].sediment.organic = _sediment[y + size_y * x][ORGANIC]
            for i in range(4):
                water[x][y].sediment.rocks[i] = _sediment[y + size_y * x][i]

            water[x][y].regolith = _regolith[x][y]

            water[x][y].velocity_x = _velocity_x[x][y]
            water[x][y].velocity_y = _velocity_y[x][y]

            # soil flow
            water_flow[x][y].resize(4, EMPTY_WATER)
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

    # timing
    cdef time_t time_start = time(NULL)

    for step in range(steps):
        print("step no. " + str(step))

        ######################################################
        # RIVER MODE
        ######################################################

        if is_river:
            # source
            water[size_x / 2][size_y / 16].height = source_height

            for x in prange(size_x, nogil=True, schedule='dynamic', chunksize=1):
                for y in range(15 * size_y / 16, size_y):
                    water[x][y] = EMPTY_WATER
   
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
                water2[x][y] = EMPTY_WATER
                for i in range(4):
                    water_flow[x][y][i] = EMPTY_WATER

        for x in range(size_x):
            for y in range(size_y):
                if water[x][y].height > 0:                    
                    height_here = heightmap[x][y] + water[x][y].height
                    acceleration = [0.0, 0.0, 0.0, 0.0]
                    out_volume_sum = 0.0
                    
                    # calculate the height differences with neighbors and flow
                    for i in range(4):
                        #if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                        height_neighbor = heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] + water[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)].height
                        delta_height = height_here - height_neighbor
                        acceleration[i] = g * delta_height / length
                        flow[x][y][i] = max(0.0, flow[x][y][i] + (delta_t * acceleration[i] * length * length))
                        out_volume_sum += delta_t * flow[x][y][i]
                            
                    # scale flow
                    scaling = 1.0
                    column_water = length * length * water[x][y].height
                    
                    if out_volume_sum > column_water and out_volume_sum > 0:
                        scaling = column_water / out_volume_sum
                        out_volume_sum = column_water
                        
                    # add to neighbors
                    for i in range(4):
                        flow[x][y][i] *= scaling

                        delta_v_x = delta_t * acceleration[i] * delta_x[i]
                        delta_v_y = delta_t * acceleration[i] * delta_y[i]

                        # move sediment
                        fraction = flow[x][y][i] * delta_t / column_water
                        water_flow[x][y][i] = speedup(water_fraction(water[x][y], fraction), delta_v_x, delta_v_y)
                    # what remains here
                    add_to_water(water2[x][y], water_fraction(water[x][y], 1 - out_volume_sum/column_water))


        # assign
        for x in range(size_x):
            for y in range(size_y):
                for i in range(4):
                    add_to_water(water2[x][y], water_flow[mod(x - delta_x[i], size_x)][mod(y - delta_y[i], size_y)][i])

                previous = water[x][y].height
                water[x][y] = water2[x][y]
                water[x][y].previous = previous
                clip_speed(water[x][y])


        ######################################################
        # FORCE EROSION
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                if water[x][y].height > 0:
                    average_height = 0.5 * (water[x][y].height + water[x][y].previous)

                    # this max function doesn't really have a physical basis, but it makes sure that small amounts of water don't erode ridiculous amounts
                    average_height = max(average_height, 0.05)

                    flow_velocity = water_velocity(x, y, size_x, size_y, flow) / (length * average_height)

                    # steepness and convexness
                    #local_shape_factor = max(0.0, 1.0 * max(0.05, get_steepness(x, y, size_x, size_y, heightmap)) + 0.0 * get_convexness(x, y, size_x, size_y, heightmap))
                    # no erosion_constant
                    water_info = erosion_constant * flow_velocity * max(0.05, get_steepness(x, y, size_x, size_y, heightmap)) * max_erosion_depth(water[x][y].height)
                    
                    # organic
                    sediment_capacity = organic_constant * water_info
                    if water[x][y].sediment.organic <= sediment_capacity:
                        # erode
                        requested_amount = 0.5 * (sediment_capacity - water[x][y].sediment.organic)
                        amount = erode_organic_amount(terrain[x][y], requested_amount)
                        heightmap[x][y] -= amount
                        water[x][y].sediment.organic += amount
                        #water[x][y] += amount
                    else:
                        # deposit
                        amount = 0.25 * (water[x][y].sediment.organic - sediment_capacity)
                        soil = new_soil(amount, 0, 0, 0, 0)
                        deposit_soil(terrain[x][y], soil)
                        heightmap[x][y] += amount
                        water[x][y].sediment.organic -= amount
                        #water[x][y] -= amount

                    # erode rocks
                    for original in range(4):
                        for sedimented in range(4):
                            sediment_capacity = max(rock_matrix_linear[original][sedimented] * water_info + rock_matrix_constant[original][sedimented], 0.0)
                            if water[x][y].sediment.rocks[sedimented] <= sediment_capacity:
                            # erode
                                requested_amount = 0.5 * (sediment_capacity - water[x][y].sediment.rocks[original])
                                amount = erode_rocks_amount(terrain[x][y], requested_amount, original)
                                heightmap[x][y] -= amount
                                water[x][y].sediment.rocks[sedimented] += amount
                                #water[x][y] += amount
                            
                    # deposit rocks
                    for rock in range(4):      
                        sediment_capacity = max(rock_matrix_linear[rock][rock] * water_info + rock_matrix_constant[rock][rock], 0.0)
                        if water[x][y].sediment.rocks[rock] > sediment_capacity:
                            # deposit
                            amount = 0.25 * (water[x][y].sediment.rocks[rock] - sediment_capacity)
                            soil = new_soil_rock(amount, rock)
                            deposit_soil(terrain[x][y], soil)
                            heightmap[x][y] += amount
                            water[x][y].sediment.rocks[rock] -= amount
                            #water[x][y] -= amount


        ######################################################
        # INERTIAL EROSION
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                if water[x][y].height > 0.0:
                    """ average_height = 0.5 * (water[x][y].height + water[x][y].previous)

                    # this max function doesn't really have a physical basis, but it makes sure that small amounts of water don't erode ridiculous amounts
                    average_height = max(average_height, 0.05) """

                    #velocity = water_velocity_vector(x, y, size_x, size_y, flow) #/ (length * average_height)
                    
                    

                    water_top = heightmap[x][y] + water[x][y].height
                    water_bottom = heightmap[x][y]
                    
                    for i in range(4):
                        velocity_in_direction = max(0.0, water[x][y].velocity_x*delta_x[i] + water[x][y].velocity_y*delta_y[i])
                        """ if x == size_x / 2 and size_y / 4 < y < 3 * size_y / 4:
                            print("velocity in y=" + str(y) + ": " + str(delta_x[i]) + ", " + str(delta_y[i]) + ": " + str(velocity_in_direction))
 """
                        height_top = heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)]
                        height_bottom = height_top - get_soil_height(terrain[x][y].topsoil)

                        # TOPSOIL
                        # calculate contact area
                        contact_top = min(water_top, height_top)
                        contact_bottom = max(water_bottom, height_bottom)
                        contact_area = max(0.0, contact_top - contact_bottom) * length

                        erosion_capacity = inertial_erosion_constant * velocity_in_direction * contact_area  # / MAX_WATER_SPEED

                        if erosion_capacity > 0.0:

                            # organic
                            sediment_capacity = organic_constant * erosion_capacity
                            #if water[x][y].sediment.organic <= sediment_capacity:
                            # erode
                            requested_amount = 0.5 * sediment_capacity
                            amount = erode_organic_amount_only_from(terrain[x][y].topsoil, requested_amount)

                            heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] -= amount
                            water[x][y].sediment.organic += amount
                                
                            # erode rocks
                            for original in range(4):
                                for sedimented in range(4):
                                    sediment_capacity = max(rock_matrix_linear[original][sedimented] * erosion_capacity + rock_matrix_constant[original][sedimented], 0.0)
                                    #if water[x][y].sediment.rocks[sedimented] <= sediment_capacity:
                                    # erode
                                    requested_amount = 0.5 * sediment_capacity
                                    amount = erode_rocks_amount_only_from(terrain[x][y].topsoil, requested_amount, original)
                                    heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] -= amount
                                    water[x][y].sediment.rocks[sedimented] += amount

                            # SUBSOIL
                            # calculate contact area
                            contact_top = min(water_top, height_bottom)
                            contact_bottom = max(water_bottom, height_bottom - get_soil_height(terrain[x][y].subsoil))
                            contact_area = max(0.0, contact_top - contact_bottom) * length

                            erosion_capacity = inertial_erosion_constant * velocity_in_direction * contact_area / MAX_WATER_SPEED

                            # organic
                            sediment_capacity = organic_constant * erosion_capacity
                            #if water[x][y].sediment.organic <= sediment_capacity:
                            # erode
                            requested_amount = 0.5 * sediment_capacity
                            amount = erode_organic_amount_only_from(terrain[x][y].subsoil, requested_amount)

                            heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] -= amount
                            water[x][y].sediment.organic += amount
                                
                            # erode rocks
                            for original in range(4):
                                for sedimented in range(4):
                                    sediment_capacity = max(rock_matrix_linear[original][sedimented] * erosion_capacity + rock_matrix_constant[original][sedimented], 0.0)
                                    #if water[x][y].sediment.rocks[sedimented] <= sediment_capacity:
                                    # erode
                                    requested_amount = 0.5 * sediment_capacity
                                    amount = erode_rocks_amount_only_from(terrain[x][y].subsoil, requested_amount, original)
                                    heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] -= amount
                                    water[x][y].sediment.rocks[sedimented] += amount

                            # slow down water
                            total_contact_volume = max(min(heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] - heightmap[x][y], water[x][y].height) * area, 0.0)

                            fraction = total_contact_volume / (water[x][y].height * area)

                            water[x][y].velocity_x *= abs(delta_x[i]) * (1.0 - fraction)
                            water[x][y].velocity_y *= abs(delta_y[i]) * (1.0 - fraction)
                                
                        


        ######################################################
        # REGOLITH EROSION
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                if water[x][y].height > 0:
                    # max regolith
                    max_regolith_thickness = max_penetration_depth
                    if water[x][y].height < max_penetration_depth:
                        max_regolith_thickness = water[x][y].height

                    # always maintain max regolith (no idea why)
                    if water[x][y].regolith < max_regolith_thickness:
                        # regolithise (not a real word)
                        amount = erode_rocks_amount(terrain[x][y], max_regolith_thickness - water[x][y].regolith, 0)
                        heightmap[x][y] -= amount
                        water[x][y].regolith += amount
                    else:
                        # deposit
                        amount = water[x][y].regolith - max_regolith_thickness
                        soil = new_soil_rock(amount, 0)
                        deposit_soil(terrain[x][y], soil)
                        heightmap[x][y] += amount
                        water[x][y].regolith -= amount


        ######################################################
        # DIFFUSE SEDIMENT
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                water2[x][y] = water[x][y]
                water2[x][y].sediment = EMPTY_SOIL

        for x in range(size_x):
            for y in range(size_y):
                # using a filter where the neighbors have weights of 0.5 and the column itself has a weight of 1.0
                # so the sum is 1.0 + 4*0.5 = 3.0
                base = 1.0 * water[x][y].height
                for i in range(4):
                    base += 0.5 * water[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)].height

                if base > 0:
                    add_to_soil(water2[x][y].sediment, soil_fraction(water[x][y].sediment, 1.0 * water[x][y].height / base))
                    #water2[x][y] -= water[x][y].sediment * (1.0 - (1.0 * water[x][y] / base))
                    for i in range(4):
                        fraction = 0.5 * water[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)].height / base
                        add_to_soil(water2[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)].sediment, soil_fraction(water[x][y].sediment, fraction))
                        #water2[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] += amount
                else:
                    add_to_soil(water2[x][y].sediment, water[x][y].sediment)
                

        # update sediment again
        for x in range(size_x):
            for y in range(size_y):
                water[x][y] = water2[x][y]


    print("done")

    # timing
    cdef time_t time_end = time(NULL)
    print("The cycle took " + str(time_end - time_start) + " seconds, or " + str((time_end - time_start) / (1.0 * steps)) + " seconds per step")

    #####################################
    # POSTPROCESSING (misusing the name, I know)

    # visually lower water which is zero in height
    cdef int neighbors_count = 0
    cdef double height_sum = 0.0
    for x in range(size_x):
        for y in range(size_y):
            if water[x][y].height <= 0.0:
                # calculate average height of neighbors
                neighbors_count = 0
                height_sum = 0.0
                for si in range(-1, 2):
                    for sj in range(-1, 2):
                        if water[mod(x+si, size_x)][mod(y+sj, size_y)].height > 0.0:
                            neighbors_count += 1
                            height_sum += water[mod(x+si, size_x)][mod(y+sj, size_y)].height + heightmap[mod(x+si, size_x)][mod(y+sj, size_y)]
                
                # set water height to average of the neighbors, if below 0
                if neighbors_count > 0:
                    water[x][y].height = min(0.0, height_sum / neighbors_count - heightmap[x][y])
                
                # if all neighbors are dry or too high
                if neighbors_count == 0 or water[x][y].height == 0.0:
                    water[x][y].height = -1.0  # just put it deep down


    # create pixel data for images
    cdef cnp.ndarray[cnp.float64_t, ndim=1] _topsoil_texture = np.zeros(shape=(4 * size_x * size_y))
    cdef double[:] topsoil_texture = _topsoil_texture
    # RGBA
    cdef double[4] color_topsoil = [1.0, 1.0, 0.0, 1.0]

    for x in range(size_x):
        for y in range(size_y):
            topsoil_texture[4 * (y + size_y * x) + 1] = max(min(0.5*water[x][y].velocity_y, 1.0), 0.0)
            topsoil_texture[4 * (y + size_y * x) + 3] = 1.0  # Alpha channel


            """ if get_soil_height(terrain[x][y].topsoil) > 0.0:
                topsoil_texture[4 * (y + size_y * x) + 1] = 1.0
                topsoil_texture[4 * (y + size_y * x) + 3] = 1.0  # Alpha channel
            if get_soil_height(terrain[x][y].subsoil) > 0.0:
                topsoil_texture[4 * (y + size_y * x) + 0] = 1.0
                topsoil_texture[4 * (y + size_y * x) + 3] = 1.0  # Alpha channel
 """
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

    _sediment = [[water[x][y].sediment.rocks[0],
                   water[x][y].sediment.rocks[1],
                   water[x][y].sediment.rocks[2],
                   water[x][y].sediment.rocks[3],
                   water[x][y].sediment.organic] for y in range(size_y) for x in range(size_x)]

    _water = [[water[x][y].height for y in range(size_y)] for x in range(size_x)]
    _previous_water = [[water[x][y].previous for y in range(size_y)] for x in range(size_x)]
    _regolith = [[water[x][y].regolith for y in range(size_y)] for x in range(size_x)]

    # return a ctuple (or tuple? tbh nobody cares, it works) of Python lists by converting a memoryview into a numpy array and then converting that into a regular array
    return (np.array(heightmap).tolist(), _water, _previous_water, _sediment, np.array(flow).tolist(), _regolith, _topsoil, _subsoil, _bedrock, np.array(topsoil_texture).tolist()) 