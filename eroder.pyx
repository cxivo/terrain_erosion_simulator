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
    Soil regolith
    Soil sediment;
    double velocity_x
    double velocity_y


##################################################
# GEOLOGY CONSTANTS

cdef Soil EMPTY_SOIL
EMPTY_SOIL.organic = 0.0
for i in range(4):
    EMPTY_SOIL.rocks[i] = 0.0

cdef Water EMPTY_WATER
EMPTY_WATER.height = 0.0
EMPTY_WATER.previous = 0.0
EMPTY_WATER.regolith = EMPTY_SOIL
EMPTY_WATER.sediment = EMPTY_SOIL
EMPTY_WATER.velocity_x = 0.0
EMPTY_WATER.velocity_y = 0.0


cdef int ORGANIC = 4  # just the id in arrays
cdef double infinity = float("inf")

cdef double MAX_TOPSOIL_HEIGHT = 1.5
cdef double LAYER_HEIGHT_TO_BECOME_ERODETHROUGH = 0.1
cdef double MIN_FRACTION_TO_ERODE = 0.1

cdef double K_EROSION = 0.1
cdef double K_IN_EROSION = 0.5
cdef double K_DEPOSITION = 0.25

cdef double MAX_WATER_SPEED = 20.0  # source: made up
cdef double WATER_MAX_DEPTH_EROSION = 0.45
cdef double g = 9.81  # source: remembered from high school
cdef double length = 1.0

cdef double organic_constant = 0.12
cdef double[4][4] rock_matrix_linear = [[0.3,   0.0,  0.0,  0.0],
                                        [0.02,  0.08, 0.0,  0.0],
                                        [0.02,  0.01, 0.04, 0.0],
                                        [0.02,  0.01, 0.01, 0.02]]
cdef double[4][4] rock_matrix_constant = [[ 0.0,  0.0,   0.0,  0.0],
                                         [-0.01, -0.1,   0.0,  0.0],
                                         [-0.02, -0.2,  -0.3,  0.0],
                                         [-0.03, -0.25, -0.4, -0.5]]

# first number is the number of rock types
cdef double[3][4] bedrock_erodability_linear = [[0.05, 0.01, 0.004, 0.002],  # soft rock
                                                [0.02, 0.005, 0.001, 0.0001],  # hard rock
                                                [0.002, 0.0005, 0.0, 0.0]]  # granite (or something)

cdef double[3][4] bedrock_erodability_constant = [[0.0, -0.25, -0.4, -0.5],  # soft rock
                                                [0.0, -0.3, -0.5, -0.7],  # hard rock
                                                [0.0, -0.5, 0.0, 0.0]]  # granite (or something)

cdef double[3] bedrock_taluses = [4.0, 10000000.0, 10000000.0]

cdef double critical_slope = 1.40
cdef double MAX_CREEP_DENOMINATOR = 0.1

cdef double ORGANIC_VEGETATION_MULIPLITER = 1.5
cdef double[4] ROCK_VEGETATION_MULIPLITER = [0.2, 0, 0, 0]
cdef double NEIGHBOR_VEGETATION_MULTIPLIER = 0.25
cdef double[4] ROCK_CONVERSION_RATE= [1, 0.2, 0.3, 0.4]

cdef double[3] BEDROCK_CONVERSION_RATE= [1, 0.3, 0]


cdef inline Soil new_soil(double organic, double rock0, double rock1, double rock2, double rock3) noexcept nogil:
    cdef Soil soil
    soil.organic = organic
    soil.rocks[0] = rock0
    soil.rocks[1] = rock1
    soil.rocks[2] = rock2
    soil.rocks[3] = rock3
    return soil


cdef Soil[3] BEDROCK_CONVERT_TO = [new_soil(0.2, 0.0, 0.0, 0.1, 0.7), new_soil(0.2, 0.0, 0.0, 0.0, 0.8), EMPTY_SOIL]


##################################################
# VARIOUS UTILS

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
    if depth >= WATER_MAX_DEPTH_EROSION:
        return 1.0
    elif 0 < depth < WATER_MAX_DEPTH_EROSION:
        return 1.0 - (WATER_MAX_DEPTH_EROSION - depth) / WATER_MAX_DEPTH_EROSION
    else:
        return 0.0



##################################################
# HELPER FUNCTIONS


cdef inline Soil new_soil_rock(double rock, int index) noexcept nogil:
    cdef Soil soil = EMPTY_SOIL
    soil.rocks[index] = rock
    return soil


cdef inline Soil new_soil_from_bedrock_type(int bedrock_type, double amount) noexcept nogil:
    if bedrock_type == 0:
        return new_soil(0, 0, 0.2 * amount, 0.3 * amount, 0.5 * amount)
    if bedrock_type == 1:
        return new_soil(0, 0, 0.1 * amount, 0.2 * amount, 0.7 * amount)
    if bedrock_type == 2:
        return new_soil(0, 0, 0.05 * amount, 0.15 * amount, 0.8 * amount)


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


# returns soil with average concentrations of all materials in a depth-deep column
cdef inline Soil get_soil_sample(Horizon& horizon, double depth) noexcept nogil:
    cdef Soil result = horizon.topsoil

    if get_soil_height(result) >= depth:
        # we have everything
        return soil_fraction(result, depth / get_soil_height(result))

    depth -= get_soil_height(result)

    if get_soil_height(horizon.subsoil) <= 0.0:
        return result

    cdef double fraction = min(1.0, depth / get_soil_height(horizon.subsoil))

    add_to_soil(result, soil_fraction(horizon.subsoil, fraction))

    return result


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

    cdef double talus = 1.25 * soil.organic
    talus += 0.7 * soil.rocks[0]
    talus += 0.8 * soil.rocks[1]
    talus += 1.0 * soil.rocks[2]
    talus += 1.0 * soil.rocks[3]

    return talus


cdef inline Soil organification(Soil soil, double vegetation, double organification_speed, double delta_t) noexcept nogil:
    cdef double change_0_org = min(delta_t * vegetation * 0.01 * organification_speed * ROCK_CONVERSION_RATE[0], soil.rocks[0])
    cdef double change_1_0 = min(delta_t * vegetation * 0.01 * organification_speed * ROCK_CONVERSION_RATE[1], soil.rocks[1])
    cdef double change_2_1 = min(delta_t * vegetation * 0.01 * organification_speed * ROCK_CONVERSION_RATE[2], soil.rocks[2])
    cdef double change_3_2 = min(delta_t * vegetation * 0.01 * organification_speed * ROCK_CONVERSION_RATE[3], soil.rocks[3])

    cdef Soil result = EMPTY_SOIL
    result.organic = soil.organic + change_0_org
    result.rocks[0] = soil.rocks[0] + change_1_0 - change_0_org
    result.rocks[1] = soil.rocks[1] + change_2_1 - change_1_0
    result.rocks[2] = soil.rocks[2] + change_3_2 - change_2_1
    result.rocks[3] = soil.rocks[3] - change_3_2

    return result

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
    water.regolith = soil_fraction(water.regolith, fraction)
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
    add_to_soil(a.regolith, b.regolith)
    add_to_soil(a.sediment, b.sediment)
    # previous water is handled differently
    a.velocity_x = a.velocity_x * velocity_multiplier + b.velocity_x * (1.0 - velocity_multiplier)
    a.velocity_y = a.velocity_y * velocity_multiplier + b.velocity_y * (1.0 - velocity_multiplier)


cdef inline Water speedup(Water water, double x, double y) noexcept nogil:
    water.velocity_x += x
    water.velocity_y += y
    return water


# MODIFIES ARGUMENT
cdef inline void slow_down(Water& water) noexcept nogil:
    cdef double scaling = 1.0
    cdef double velocity = sqrt(water.velocity_x**2 + water.velocity_y**2)
    cdef double resistance = 0.25 * (velocity / MAX_WATER_SPEED)**2

    #if velocity > MAX_WATER_SPEED:
    #    scaling = MAX_WATER_SPEED / velocity
    water.velocity_x *= max(1 - resistance, 0)
    water.velocity_y *= max(1 - resistance, 0)


##################################################
# EROSION AND DEPOSITION


# MODIFIES ARGUMENT
cdef void remove_soil(Horizon& horizon, double amount) noexcept nogil:
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
    
    # first non-empty
    cdef int index = 0
    while horizon.bedrock[index].height <= 0.0:
        index += 1

    # remove the rest from the bedrock
    horizon.bedrock[index].height -= amount - removed



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
    cdef double amount_remove = 0.0
    cdef double removed = 0.0
    requested_amount = max(0.0, requested_amount)

    if get_soil_height(horizon.topsoil) > 0.0:
        amount_remove = min(horizon.topsoil.organic, requested_amount)
        requested_amount -= amount_remove
        amount_remove *= max(MIN_FRACTION_TO_ERODE, horizon.topsoil.organic / get_soil_height(horizon.topsoil))
        horizon.topsoil.organic -= amount_remove
        removed = amount_remove
        # try eroding from the lower layer as well
        requested_amount *= max(0.0, 1 - (get_soil_height(horizon.topsoil) / LAYER_HEIGHT_TO_BECOME_ERODETHROUGH))
        
    if get_soil_height(horizon.subsoil) > 0.0 and requested_amount > 0.0:
        amount_remove = min(horizon.subsoil.organic, requested_amount)
        requested_amount -= amount_remove
        amount_remove *= max(MIN_FRACTION_TO_ERODE, horizon.subsoil.organic / get_soil_height(horizon.subsoil))
        horizon.subsoil.organic -= amount_remove     

    # no need to erode rocks, they aren't organic
    return amount_remove + removed


# MODIFIES ARGUMENT
cdef double erode_rocks_amount(Horizon& horizon, double water_info, int type_result) noexcept nogil:
    cdef double amount_remove
    cdef double removed = 0.0
    cdef double requested_amount = 0.0
    cdef double requested_residue = 0.0
    cdef int i = 0


    for i in range(4):
        requested_amount = K_EROSION * max(rock_matrix_linear[i][type_result] * water_info + rock_matrix_constant[i][type_result], 0.0)
        if horizon.topsoil.rocks[i] > 0.0 and requested_amount > 0.0:
            amount_remove = min(horizon.topsoil.rocks[i], requested_amount)
            requested_amount -= amount_remove
            amount_remove *= max(MIN_FRACTION_TO_ERODE, horizon.topsoil.rocks[i] / get_soil_height(horizon.topsoil))
            horizon.topsoil.rocks[i] -= amount_remove
            removed += amount_remove

            # try eroding from the lower layer as well
            requested_amount *= max(0.0, 1 - (get_soil_height(horizon.topsoil) / LAYER_HEIGHT_TO_BECOME_ERODETHROUGH))


        if horizon.subsoil.rocks[i] > 0.0 and requested_amount > 0.0:
            amount_remove = min(horizon.subsoil.rocks[i], requested_amount)
            requested_amount -= amount_remove
            amount_remove *= max(MIN_FRACTION_TO_ERODE, horizon.subsoil.rocks[i] / get_soil_height(horizon.subsoil))
            horizon.subsoil.rocks[i] -= amount_remove
            removed += amount_remove

            # try eroding from the lower layer as well
            requested_amount *= max(0.0, 1 - (get_soil_height(horizon.subsoil) / LAYER_HEIGHT_TO_BECOME_ERODETHROUGH))

        # how much will it try to erode rocks
        requested_residue += requested_amount
    
    # bedrock
    for i in range(horizon.bedrock.size()):
        if horizon.bedrock[i].height <= 0.0:
            continue

        rock_type = horizon.bedrock[i].type
        requested_amount = max(bedrock_erodability_linear[rock_type][type_result] * requested_residue + bedrock_erodability_constant[rock_type][type_result], 0.0)
        if requested_amount > 0.0:
            amount_remove = min(horizon.bedrock[i].height, requested_amount)
            horizon.bedrock[i].height -= amount_remove
            removed += amount_remove
            break

    return removed


# MODIFIES ARGUMENT
cdef Soil erode_regolith(Horizon& horizon, double requested_amount) noexcept nogil:
    cdef double amount_remove_org = 0.0, amount_remove_rock = 0.0
    cdef double removed_org = 0.0, removed_rock = 0.0
    requested_amount = max(0.0, requested_amount)

    if get_soil_height(horizon.topsoil) > 0.0:
        amount_remove_org = min(horizon.topsoil.organic, 0.5 * requested_amount)
        amount_remove_rock = min(horizon.topsoil.rocks[0], 0.5 * requested_amount)
        requested_amount -= amount_remove_org + amount_remove_rock
        amount_remove_org *= max(MIN_FRACTION_TO_ERODE, horizon.topsoil.organic / get_soil_height(horizon.topsoil))
        amount_remove_rock *= max(MIN_FRACTION_TO_ERODE, horizon.topsoil.rocks[0] / get_soil_height(horizon.topsoil))
        horizon.topsoil.organic -= amount_remove_org
        horizon.topsoil.rocks[0] -= amount_remove_rock
        removed_org = amount_remove_org 
        removed_rock = amount_remove_rock
        # try eroding from the lower layer as well
        requested_amount *= max(0.0, 1 - (get_soil_height(horizon.topsoil) / LAYER_HEIGHT_TO_BECOME_ERODETHROUGH))
        
    if get_soil_height(horizon.subsoil) > 0.0 and requested_amount > 0.0:
        amount_remove_org = min(horizon.subsoil.organic, 0.5 * requested_amount)
        amount_remove_rock = min(horizon.subsoil.rocks[0], 0.5 * requested_amount)
        requested_amount -= amount_remove_org + amount_remove_rock
        amount_remove_org *= max(MIN_FRACTION_TO_ERODE, horizon.subsoil.organic / get_soil_height(horizon.subsoil))
        amount_remove_rock *= max(MIN_FRACTION_TO_ERODE, horizon.subsoil.rocks[0] / get_soil_height(horizon.subsoil))
        horizon.subsoil.organic -= amount_remove_org
        horizon.subsoil.rocks[0] -= amount_remove_rock
        

    # no need to erode rocks, they can't get regolith layer
    return new_soil(amount_remove_org + removed_org, amount_remove_rock + removed_rock, 0, 0, 0)


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


# this function exists because it some-fricking-how interfered with Gravitational erosion and I couldn't find a fix
# MODIFIES ARGUMENTS
cdef void gravitational_erosion(int size_x, int size_y, double delta_t, double downhill_creep_constant, vector[vector[Horizon]]& terrain, double[:, :]& heightmap, double[:, :]& vegetation) noexcept nogil:
    cdef vector[vector[vector[vector[Soil]]]] soil_flow
    cdef vector[vector[double]] heightmap2
    cdef int x, y, si, sj, i, index
    cdef double fraction, total_quantity, slope, denominator, q, to_distribute, height_fraction, part, delta_h_sum
    cdef Soil soil
    cdef double[3][3] neighbors_delta_height

    heightmap2.resize(size_x)
    soil_flow.resize(size_x)

    for x in range(size_x):
        heightmap2.at(x).resize(size_y, 0.0)
        soil_flow.at(x).resize(size_y)
        for y in range(size_y):
            soil_flow[x][y].resize(3)
            for si in range(3):
                soil_flow[x][y][si].resize(3, EMPTY_SOIL)


    for x in range(size_x):
        for y in range(size_y):
            # clear flows
            heightmap2[x][y] = 0.0
            for si in range(-1, 2):
                for sj in range(-1, 2):
                    soil_flow[x][y][si + 1][sj + 1] = EMPTY_SOIL

            # talus angle will be the smaller of the 2 soils
            talus = get_soil_talus(terrain[x][y].topsoil)
            if get_soil_talus(terrain[x][y].subsoil) * 1.25 < talus:
                talus = get_soil_talus(terrain[x][y].subsoil) * 1.25



            # which neighbors will get a piece of our terrain
            soil = terrain[x][y].topsoil
            for i in range(2):

                delta_h_sum = 0.0
                max_delta = 0.0

                if get_soil_height(soil) > 0.0:
                    neighbors_delta_height = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] 
                    for si in range(-1, 2):
                        for sj in range(-1, 2):
                            if si != 0 or sj != 0:
                                delta = heightmap[x][y] - heightmap[mod(x + si, size_x)][mod(y + sj, size_y)]

                                # if should erode
                                if delta / (length * sqrt(si**2 + sj**2)) >= talus:
                                    # largest difference
                                    if delta > max_delta:
                                        max_delta = delta

                                    # save for later reference
                                    neighbors_delta_height[si + 1][sj + 1] = delta
                                    delta_h_sum += delta
                                    
                    
                    to_distribute = min(0.5 * max_delta * random(0.9, 1.0), get_soil_height(soil))

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
                    soil = terrain[x][y].subsoil


            # BEDROCK only if there is nothing above us
            if get_soil_height(terrain[x][y].topsoil) + get_soil_height(terrain[x][y].subsoil) <= 0.0:
                talus = infinity
                for i in range(terrain[x][y].bedrock.size()):
                    if bedrock_taluses[terrain[x][y].bedrock[i].type] < talus and terrain[x][y].bedrock[i].height > 0.0:
                        talus = bedrock_taluses[terrain[x][y].bedrock[i].type]
                
                delta_h_sum = 0.0
                max_delta = 0.0

                # first nonempty
                index = 0
                while terrain[x][y].bedrock[index].height <= 0.0:
                    index += 1

            
                neighbors_delta_height = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] 
                for si in range(-1, 2):
                    for sj in range(-1, 2):
                        if si != 0 or sj != 0:
                            delta = heightmap[x][y] - heightmap[mod(x + si, size_x)][mod(y + sj, size_y)]

                            # if should erode
                            if delta / (length * sqrt(si**2 + sj**2)) >= talus:
                                # largest difference
                                if delta > max_delta:
                                    max_delta = delta

                                # save for later reference
                                neighbors_delta_height[si + 1][sj + 1] = delta
                                delta_h_sum += delta
                                
                
                to_distribute = min(0.5 * max_delta * random(0.9, 1.0), terrain[x][y].bedrock[index].height)

                # how much will flow OUT of this cell later
                # misusing this array
                heightmap2[x][y] = to_distribute

                # assign flows
                if delta_h_sum > 0:
                    for si in range(-1, 2):
                        for sj in range(-1, 2):
                            if si != 0 or sj != 0:
                                part = to_distribute * neighbors_delta_height[si + 1][sj + 1] / delta_h_sum
                                soil_flow[x][y][si + 1][sj + 1] = new_soil_from_bedrock_type(index, part)


    # assign flows
    for x in range(size_x):
        for y in range(size_y):
            # remove soil from this column
            remove_soil(terrain[x][y], heightmap2[x][y])
            heightmap[x][y] -= heightmap2[x][y]

            # kills vegetation
            if heightmap2[x][y] > 0.0:
                vegetation[x][y] = 0.0

            # deposit soil from neighbors
            for si in range(-1, 2):
                for sj in range(-1, 2):
                    if si != 0 or sj != 0:
                        deposit_soil(terrain[x][y], soil_flow[mod(x + si, size_x)][mod(y + sj, size_y)][1 - si][1 - sj])
                        soil_height = get_soil_height(soil_flow[mod(x + si, size_x)][mod(y + sj, size_y)][1 - si][1 - sj])
                        heightmap[x][y] += soil_height

                        # kills vegetation
                        if soil_height > 0.0:
                            vegetation[x][y] = 0.0


# this function exists because it some-fricking-how interfered with Gravitational erosion and I couldn't find a fix
# MODIFIES ARGUMENTS
cdef void downhill_creep(int size_x, int size_y, double delta_t, double downhill_creep_constant, vector[vector[Horizon]]& terrain, double[:, :]& heightmap) noexcept nogil:
    cdef vector[vector[vector[vector[Soil]]]] soil_flow
    cdef vector[vector[double]] heightmap2
    cdef int x, y, si, sj
    cdef double fraction, total_quantity, slope, denominator, q, to_distribute, height_fraction
    cdef Soil soil
    cdef double[3][3] neighbors_quantity

    heightmap2.resize(size_x)
    soil_flow.resize(size_x)

    for x in range(size_x):
        heightmap2.at(x).resize(size_y, 0.0)
        soil_flow.at(x).resize(size_y)
        for y in range(size_y):
            soil_flow[x][y].resize(3)
            for si in range(3):
                soil_flow[x][y][si].resize(3, EMPTY_SOIL)


    for x in range(size_x):
        for y in range(size_y):
            # clear flows
            heightmap2[x][y] = 0.0
            for si in range(-1, 2):
                for sj in range(-1, 2):
                    soil_flow[x][y][si + 1][sj + 1] = EMPTY_SOIL

            # which neighbors will get a piece of our terrain
            soil = terrain[x][y].topsoil
            for i in range(2):
                
                total_quantity = 0.0

                if get_soil_height(soil) > 0.0:
                    neighbors_quantity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] 
                    for si in range(-1, 2):
                        for sj in range(-1, 2):
                            if si != 0 or sj != 0:
                                slope = heightmap[x][y] - heightmap[mod(x + si, size_x)][mod(y + sj, size_y)]
                                if slope > 0.0: 
                                    if si != 0 and sj != 0:
                                        slope /= sqrt(2) * length
                                    else:
                                        slope /= length

                                    denominator = 1 - (slope / critical_slope)**2

                                    # if it's less than 0 it's too steep, should've gotten eroded by gravitational erosion anyway
                                    denominator = max(denominator, MAX_CREEP_DENOMINATOR)

                                    q = downhill_creep_constant * slope / denominator

                                    # save for later reference
                                    neighbors_quantity[si + 1][sj + 1] = q
                                    total_quantity += q
                                
                    to_distribute = min(delta_t * 0.5 * total_quantity * soil.organic / get_soil_height(soil), get_soil_height(soil))

                    height_fraction = to_distribute / get_soil_height(soil)

                    # how much will flow OUT of this cell later
                    # misusing this array
                    heightmap2[x][y] = to_distribute

                    # assign flows
                    if to_distribute > 0:
                        for si in range(-1, 2):
                            for sj in range(-1, 2):
                                if si != 0 or sj != 0:
                                    fraction = height_fraction * neighbors_quantity[si + 1][sj + 1] / total_quantity
                                    soil_flow[x][y][si + 1][sj + 1] = soil_fraction(soil, fraction)
                    
                    # erode only the topsoil
                    break
                else:
                    soil = terrain[x][y].subsoil

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


##################################################
##################################################
cpdef erode(unsigned int steps, double delta_t, double erosion_constant, double max_penetration_depth, double inertial_erosion_constant, double downhill_creep_constant, double vegetation_spread_speed, double vegetation_base, double organification_speed, double soilification_speed, bint is_river, double source_height, unsigned int size_x, unsigned int size_y, list _heightmap, list _water, list _previous_water, list _sediment, list _flow, list _regolith, list _topsoil, list _subsoil, list _bedrock, list _bedrock_types, list _velocity_x, list _velocity_y, list _wet, list _vegetation):
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
    cdef double[:, ::1] wet = np.array(_wet, dtype=np.float64)
    cdef double[:, ::1] vegetation = np.array(_vegetation, dtype=np.float64)


    cdef unsigned int step = 0
    cdef int[4][3] neighbors = [[1, 0, 0], [0, 1, 1], [-1, 0, 2], [0, -1, 3]]
    cdef double area = length * length
    cdef int[4] delta_x = [ 1, 0, -1, 0 ]
    cdef int[4] delta_y = [ 0, 1, 0, -1 ]

    

    cdef vector[vector[Water]] water
    cdef vector[vector[Water]] water2
    cdef vector[vector[Horizon]] terrain
    cdef vector[vector[vector[Water]]] water_flow
    cdef vector[vector[vector[vector[Soil]]]] soil_flow
    cdef vector[vector[double]] heightmap2
    cdef vector[vector[double]] vegetation2
    """ cdef double[4] acceleration = [0.0, 0.0, 0.0, 0.0]
    cdef double delta_height
    cdef double scaling
    cdef double flow_velocity
    cdef double local_shape_factor
    cdef double sediment_capacity
    cdef double out_volume_sum
    cdef double[3][3] neighbors_delta_height"""
    cdef Soil soil 

    print("starting erosion...") 

    # initialize vectors
    water.resize(size_x)
    water2.resize(size_x)
    terrain.resize(size_x)
    soil_flow.resize(size_x)
    water_flow.resize(size_x)
    heightmap2.resize(size_x)
    vegetation2.resize(size_x)

    for x in range(size_x):
        water.at(x).resize(size_y, EMPTY_WATER)
        water2.at(x).resize(size_y, EMPTY_WATER)
        soil_flow.at(x).resize(size_y)
        water_flow.at(x).resize(size_y)
        terrain.at(x).resize(size_y)
        heightmap2.at(x).resize(size_y, 0.0)
        vegetation2.at(x).resize(size_y, 0.0)
        for y in range(size_y):
            # water 
            water[x][y].height = _water[x][y]
            if water[x][y].height < 0.0:
                water[x][y].height = 0.0

            water[x][y].previous = _previous_water[x][y]

            water[x][y].sediment.organic = _sediment[x + size_x * y][ORGANIC]
            for i in range(4):
                water[x][y].sediment.rocks[i] = _sediment[x + size_x * y][i]

            water[x][y].regolith.organic = _regolith[x + size_x * y][0]
            water[x][y].regolith.rocks[0] = _regolith[x + size_x * y][1]
            for i in range(1, 4):
                water[x][y].regolith.rocks[i] = 0.0

            water[x][y].velocity_x = _velocity_x[x][y]
            water[x][y].velocity_y = _velocity_y[x][y]

            # soil flow
            water_flow[x][y].resize(4, EMPTY_WATER)
            soil_flow[x][y].resize(3)
            for i in range(3):
                soil_flow[x][y][i].resize(3)            

            # init terrain
            terrain[x][y].topsoil.organic = _topsoil[x + size_x * y][ORGANIC]
            for i in range(4):
                terrain[x][y].topsoil.rocks[i] = _topsoil[x + size_x * y][i]

            terrain[x][y].subsoil.organic = _subsoil[x + size_x * y][ORGANIC]
            for i in range(4):
                terrain[x][y].subsoil.rocks[i] = _subsoil[x + size_x * y][i]

            vector_size = len(_bedrock[x + size_x * y])
            terrain[x][y].bedrock.resize(vector_size)
            for i in range(vector_size):
                terrain[x][y].bedrock[i].type = round(_bedrock_types[i])
                terrain[x][y].bedrock[i].height = _bedrock[x + size_x * y][i]

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
        # DOWNHILL CREEP
        ######################################################

        downhill_creep(size_x, size_y, delta_t, downhill_creep_constant, terrain, heightmap)

        ######################################################
        # GRAVITATIONAL EROSION
        ######################################################

        gravitational_erosion(size_x, size_y, delta_t, downhill_creep_constant, terrain, heightmap, vegetation)

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

                        velocity_in_direction = max(0.0, water[x][y].velocity_x*delta_x[i] + water[x][y].velocity_y*delta_y[i])
                        flow[x][y][i] = max(0.0, flow[x][y][i] + ((delta_t * acceleration[i]) * length * length))
                        out_volume_sum += delta_t * flow[x][y][i]

                        """ if velocity_in_direction > 0:
                            print(velocity_in_direction) """
                            
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
                slow_down(water[x][y])


        ######################################################
        # FORCE EROSION
        ######################################################
        for x in range(size_x):
            for y in range(size_y):
                if water[x][y].height > 0:
                    average_height = 0.5 * (water[x][y].height + water[x][y].previous)

                    # this max function doesn't really have a physical basis, but it makes sure that small amounts of water don't erode ridiculous amounts
                    average_height = max(average_height, 0.05) # 0.05

                    flow_velocity = water_velocity(x, y, size_x, size_y, flow) / (length * average_height)

                    # steepness and convexness
                    #local_shape_factor = max(0.0, 1.0 * max(0.05, get_steepness(x, y, size_x, size_y, heightmap)) + 0.0 * get_convexness(x, y, size_x, size_y, heightmap))
                    # no erosion_constant
                    water_info = erosion_constant * flow_velocity * max(0.05, get_steepness(x, y, size_x, size_y, heightmap)) * max_erosion_depth(water[x][y].height)
                    
                    # organic
                    sediment_capacity = organic_constant * water_info
                    if water[x][y].sediment.organic <= sediment_capacity:
                        # erode
                        requested_amount = K_EROSION * (sediment_capacity - water[x][y].sediment.organic)
                        amount = erode_organic_amount(terrain[x][y], requested_amount)
                        heightmap[x][y] -= amount
                        water[x][y].sediment.organic += amount
                        #water[x][y] += amount
                    else:
                        # deposit
                        amount = K_DEPOSITION * (water[x][y].sediment.organic - sediment_capacity)
                        soil = new_soil(amount, 0, 0, 0, 0)
                        deposit_soil(terrain[x][y], soil)
                        heightmap[x][y] += amount
                        water[x][y].sediment.organic -= amount
                        #water[x][y] -= amount

                    # erode rocks
                    for rock in range(4):                        
                        # erode
                        amount = erode_rocks_amount(terrain[x][y], water_info, rock)
                        heightmap[x][y] -= amount
                        water[x][y].sediment.rocks[rock] += amount
                        #water[x][y] += amount
                            
                    # deposit rocks
                    for rock in range(4):      
                        sediment_capacity = max(rock_matrix_linear[rock][rock] * water_info + rock_matrix_constant[rock][rock], 0.0)
                        if water[x][y].sediment.rocks[rock] > sediment_capacity:

                            # deposit
                            amount = K_DEPOSITION * (water[x][y].sediment.rocks[rock] - sediment_capacity)
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
                            requested_amount = K_IN_EROSION* sediment_capacity
                            amount = erode_organic_amount_only_from(terrain[x][y].topsoil, requested_amount)

                            heightmap[mod(x + delta_x[i], size_x)][mod(y + delta_y[i], size_y)] -= amount
                            water[x][y].sediment.organic += amount
                                
                            # erode rocks
                            for original in range(4):
                                for sedimented in range(4):
                                    sediment_capacity = max(rock_matrix_linear[original][sedimented] * erosion_capacity + rock_matrix_constant[original][sedimented], 0.0)
                                    #if water[x][y].sediment.rocks[sedimented] <= sediment_capacity:
                                    # erode
                                    requested_amount = K_IN_EROSION * sediment_capacity
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
        # DISSOLUTION/REGOLITH EROSION
        ######################################################
        if max_penetration_depth > 0.0:
            for x in range(size_x):
                for y in range(size_y):
                    if water[x][y].height > 0:
                        # max regolith
                        max_regolith_thickness = max_penetration_depth
                        if water[x][y].height < max_penetration_depth:
                            max_regolith_thickness = water[x][y].height

                        # always maintain max regolith (no idea why)
                        if get_soil_height(water[x][y].regolith) < max_regolith_thickness:
                            # regolithise (not a real word)
                            amount = erode_regolith(terrain[x][y], max_regolith_thickness - get_soil_height(water[x][y].regolith))
                            heightmap[x][y] -= get_soil_height(amount)
                            add_to_soil(water[x][y].regolith, amount)
                        else:
                            # deposit
                            amount = get_soil_height(water[x][y].regolith) - max_regolith_thickness
                            soil = soil_fraction(water[x][y].regolith, amount / get_soil_height(water[x][y].regolith))
                            deposit_soil(terrain[x][y], soil)
                            heightmap[x][y] += amount
                            water[x][y].regolith = soil_fraction(water[x][y].regolith, 1 - amount / get_soil_height(water[x][y].regolith))


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

           
        ######################################################
        # WETNESS
        ######################################################

        for x in range(size_x):
            for y in range(size_y):
                if water[x][y].height > 0.0:
                    wet[x][y] += delta_t * water[x][y].height
                else:
                    wet[x][y] *= 0.5**delta_t 

        ######################################################
        # VEGETATION
        ######################################################

        for x in range(size_x):
            for y in range(size_y):
                # dying
                if wet[x][y] > 1.0:
                    # death from floods
                    vegetation[x][y] = 0.0
                elif terrain[x][y].topsoil.organic + terrain[x][y].subsoil.organic == 0.0:
                    # death from no soil
                    vegetation[x][y] = 0.0
                else:
                # growing
                    #if vegetation[x][y] == 0.0:

                    # count neighbors
                    neighbor_vegetation = 0.0
                    for si in range(-1, 2):
                        for sj in range(-1, 2):
                            neighbor_vegetation += vegetation[mod(x + si, size_x)][mod(y + sj, size_y)]

                    # soil conditions
                    average_soil_shallow = get_soil_sample(terrain[x][y], 0.25)
                    average_soil_deep = get_soil_sample(terrain[x][y], 2.0)

                    # set to 1 if greater than 1
                    if get_soil_height(average_soil_shallow) > 1:
                        average_soil_shallow = normalized(average_soil_shallow)

                    if get_soil_height(average_soil_deep) > 1:
                        average_soil_deep = normalized(average_soil_deep)

                    rocks_shallow = 0.0
                    for i in range(4):
                        rocks_shallow += ROCK_VEGETATION_MULIPLITER[i] * average_soil_shallow.rocks[i]
                        
                    rocks_deep = 0.0
                    for i in range(4):
                        rocks_deep += ROCK_VEGETATION_MULIPLITER[i] * average_soil_deep.rocks[i]

                    # how good the soil is for the plants
                    soil_conditions = (1 - vegetation[x][y]) * (ORGANIC_VEGETATION_MULIPLITER * average_soil_shallow.organic + rocks_shallow) + vegetation[x][y] * (ORGANIC_VEGETATION_MULIPLITER * average_soil_deep.organic + rocks_deep) - vegetation_base

                    vegetation2[x][y] = min(max(0.0, vegetation[x][y] + random(0, 1) * delta_t * vegetation_spread_speed * (soil_conditions - 0.8 * vegetation[x][y]) * (1 + NEIGHBOR_VEGETATION_MULTIPLIER * neighbor_vegetation)), 1.0)

        for x in range(size_x):
            for y in range(size_y):
                vegetation[x][y] = vegetation2[x][y]

        ######################################################
        # ROCK CONVERSION
        ######################################################


        for x in range(size_x):
            for y in range(size_y):
                if vegetation[x][y] > 0.0:
                    # change soil rocks into organic or break them down
                    terrain[x][y].topsoil = organification(terrain[x][y].topsoil, vegetation[x][y], organification_speed, delta_t)
                    terrain[x][y].subsoil = organification(terrain[x][y].subsoil, vegetation[x][y], organification_speed, delta_t)

                    # change bedrock into soil
                    bedrock_depth = get_soil_height(terrain[x][y].topsoil) + get_soil_height(terrain[x][y].subsoil)
                    if bedrock_depth <= 4.0:
                        # first nonempty
                        index = 0
                        while terrain[x][y].bedrock[index].height <= 0.0:
                            index += 1

                        amount = min(delta_t * vegetation[x][y] * 0.01 * soilification_speed * BEDROCK_CONVERSION_RATE[terrain[x][y].bedrock[index].type], terrain[x][y].bedrock[index].height)
                        soil = soil_fraction(BEDROCK_CONVERT_TO[terrain[x][y].bedrock[index].type], amount)
                        
                        # transfer soil
                        terrain[x][y].bedrock[index].height -= amount
                        add_to_soil(terrain[x][y].subsoil, soil)
                    
                    

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
    #cdef cnp.ndarray[cnp.float64_t, ndim=1] _topsoil_texture = 
    cdef double[:] texture_type = np.zeros(shape=(4 * size_x * size_y))

    cdef double[:] texture_organic = np.zeros(shape=(4 * size_x * size_y))
    cdef double[:] texture_rock0 = np.zeros(shape=(4 * size_x * size_y))
    cdef double[:] texture_rock1 = np.zeros(shape=(4 * size_x * size_y))
    cdef double[:] texture_rock2 = np.zeros(shape=(4 * size_x * size_y))
    cdef double[:] texture_rock3 = np.zeros(shape=(4 * size_x * size_y))
    cdef double[:] texture_bedrock0 = np.zeros(shape=(4 * size_x * size_y))
    cdef double[:] texture_bedrock1 = np.zeros(shape=(4 * size_x * size_y))
    cdef double[:] texture_bedrock2 = np.zeros(shape=(4 * size_x * size_y))

    cdef double[:] texture_vegetation = np.zeros(shape=(4 * size_x * size_y))

    cdef double[:] texture_water = np.zeros(shape=(4 * size_x * size_y))

    cdef double[:] texture_velocity = np.zeros(shape=(4 * size_x * size_y))
    
    # RGBA
    cdef double[4] color_topsoil = [1.0, 1.0, 0.0, 1.0]

    for x in range(size_x):
        for y in range(size_y):
            # velocity
            texture_velocity[4 * (x + size_x * y)] = max(min(0.5*sqrt(water[x][y].velocity_y**2 + water[x][y].velocity_x**2), 1.0), 0.0)
            texture_velocity[4 * (x + size_x * y) + 1] = max(min(0.5*water[x][y].velocity_y, 1.0), 0.0)
            texture_velocity[4 * (x + size_x * y) + 2] = max(min(-0.5*water[x][y].velocity_y, 1.0), 0.0)
            texture_velocity[4 * (x + size_x * y) + 3] = 1.0  # Alpha channel

            # topsoil texture
            texture_type[4 * (x + size_x * y) + 1] = min(1.0, get_soil_height(terrain[x][y].topsoil))
            texture_type[4 * (x + size_x * y) + 0] = min(1.0, get_soil_height(terrain[x][y].subsoil))
            texture_type[4 * (x + size_x * y) + 3] = 1.0  # Alpha channel

                #for i in range(4):
                #    topsoil_texture[4 * (x + size_x * y) + i] = color_topsoil[i]


            # material textures
            render_soil = EMPTY_SOIL
            multiplier = 1.0

            # topsoil
            height = get_soil_height(terrain[x][y].topsoil)
            
            if height > 0.0:
                render_soil.organic = terrain[x][y].topsoil.organic / height
                for i in range(4):
                    render_soil.rocks[i] = terrain[x][y].topsoil.rocks[i] / height
                # how transparent
                multiplier *= max(0.0, 1 - (height / LAYER_HEIGHT_TO_BECOME_ERODETHROUGH))

            # subsoil
            height = get_soil_height(terrain[x][y].subsoil)
        
            if height > 0.0 and multiplier > 0.0:
                render_soil.organic += terrain[x][y].subsoil.organic / height
                for i in range(4):
                    render_soil.rocks[i] += multiplier * terrain[x][y].subsoil.rocks[i] / height
                # how transparent
                multiplier *= max(0.0, 1 - (height / LAYER_HEIGHT_TO_BECOME_ERODETHROUGH))

            render_soil = normalized(render_soil)
            render_soil = soil_fraction(render_soil, 1 - multiplier)

            # write to textures
            texture_organic[4 * (x + size_x * y)] = render_soil.organic
            texture_organic[4 * (x + size_x * y) + 1] = render_soil.organic
            texture_organic[4 * (x + size_x * y) + 2] = render_soil.organic
            texture_organic[4 * (x + size_x * y) + 3] = 1.0

            
            texture_rock0[4 * (x + size_x * y)] = render_soil.rocks[0]
            texture_rock0[4 * (x + size_x * y) + 1] = render_soil.rocks[0]
            texture_rock0[4 * (x + size_x * y) + 2] = render_soil.rocks[0]
            texture_rock0[4 * (x + size_x * y) + 3] = 1.0

            texture_rock1[4 * (x + size_x * y)] = render_soil.rocks[1]
            texture_rock1[4 * (x + size_x * y) + 1] = render_soil.rocks[1]
            texture_rock1[4 * (x + size_x * y) + 2] = render_soil.rocks[1]
            texture_rock1[4 * (x + size_x * y) + 3] = 1.0

            texture_rock2[4 * (x + size_x * y)] = render_soil.rocks[2]
            texture_rock2[4 * (x + size_x * y) + 1] = render_soil.rocks[2]
            texture_rock2[4 * (x + size_x * y) + 2] = render_soil.rocks[2]
            texture_rock2[4 * (x + size_x * y) + 3] = 1.0

            texture_rock3[4 * (x + size_x * y)] = render_soil.rocks[3]
            texture_rock3[4 * (x + size_x * y) + 1] = render_soil.rocks[3]
            texture_rock3[4 * (x + size_x * y) + 2] = render_soil.rocks[3]
            texture_rock3[4 * (x + size_x * y) + 3] = 1.0

            # water and wetness
            if water[x][y].height > 0.0:
                texture_water[4 * (x + size_x * y)] = min(water[x][y].height, 1.0)
                texture_water[4 * (x + size_x * y) + 1] = min(water[x][y].height, 1.0)
                texture_water[4 * (x + size_x * y) + 2] = min(water[x][y].height, 1.0)
            texture_water[4 * (x + size_x * y) + 3] = 1.0



            # bedrock
            if multiplier > 0.0:
                # first non-empty
                i = 0
                while terrain[x][y].bedrock[i].height <= 0.0:
                    i += 1

                if i == 0:
                    texture_bedrock0[4 * (x + size_x * y)] = multiplier
                    texture_bedrock0[4 * (x + size_x * y) + 1] = multiplier
                    texture_bedrock0[4 * (x + size_x * y) + 2] = multiplier
                elif i == 1:
                    texture_bedrock1[4 * (x + size_x * y)] = multiplier
                    texture_bedrock1[4 * (x + size_x * y) + 1] = multiplier
                    texture_bedrock1[4 * (x + size_x * y) + 2] = multiplier
                elif i == 2:
                    texture_bedrock2[4 * (x + size_x * y)] = multiplier
                    texture_bedrock2[4 * (x + size_x * y) + 1] = multiplier
                    texture_bedrock2[4 * (x + size_x * y) + 2] = multiplier
            texture_bedrock0[4 * (x + size_x * y) + 3] = 1.0  # transparency
            texture_bedrock1[4 * (x + size_x * y) + 3] = 1.0  # transparency
            texture_bedrock2[4 * (x + size_x * y) + 3] = 1.0  # transparency


            # vegetation
            texture_vegetation[4 * (x + size_x * y)] = vegetation[x][y]
            texture_vegetation[4 * (x + size_x * y) + 1] = vegetation[x][y]
            texture_vegetation[4 * (x + size_x * y) + 2] = vegetation[x][y]
            texture_vegetation[4 * (x + size_x * y) + 3] = 1.0
        


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

    _bedrock = [[terrain[x][y].bedrock[i].height
                 for i in range(len(terrain[x][y].bedrock))] for y in range(size_y) for x in range(size_x)]

    _sediment = [[water[x][y].sediment.rocks[0],
                   water[x][y].sediment.rocks[1],
                   water[x][y].sediment.rocks[2],
                   water[x][y].sediment.rocks[3],
                   water[x][y].sediment.organic] for y in range(size_y) for x in range(size_x)]

    _water = [[water[x][y].height for y in range(size_y)] for x in range(size_x)]
    _previous_water = [[water[x][y].previous for y in range(size_y)] for x in range(size_x)]
    _regolith = [[water[x][y].regolith.organic, water[x][y].regolith.rocks[0]] for y in range(size_y) for x in range(size_x)]

    # return a ctuple (or tuple? tbh nobody cares, it works) of Python lists by converting a memoryview into a numpy array and then converting that into a regular array
    return (np.array(heightmap).tolist(),
    _water,
    _previous_water, 
    _sediment, 
    np.array(flow).tolist(), 
    _regolith, 
    _topsoil, 
    _subsoil, 
    _bedrock, 
    _bedrock_types, 
    np.array(wet).tolist(),
    np.array(vegetation).tolist(), 
    np.array(texture_type).tolist(), 
    np.array(texture_velocity).tolist(), 
    np.array(texture_organic).tolist(), 
    (np.array(texture_rock0).tolist(), np.array(texture_rock1).tolist(), np.array(texture_rock2).tolist(), np.array(texture_rock3).tolist()), 
    (np.array(texture_bedrock0).tolist(), np.array(texture_bedrock1).tolist(), np.array(texture_bedrock2).tolist()),
    np.array(texture_water).tolist(),
    np.array(texture_vegetation).tolist()) 