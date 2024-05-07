# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "Terrain erosion simulator",
    "author" : "cxivo",
    "description" : "",
    "blender" : (4, 0, 0),
    "version" : (0, 0, 2),
    "location": "View3D > Add > Mesh > New Terrain",
    "description": "Adds a new Terrain",
    "warning" : "",
    "category" : "Generic"
}


import bpy
import math
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector, noise
from .eroder import erode


def add_object(self, context):
    # creates a plane
    verts = [
        Vector(
            (self.scale * (0.5 + x - self.size_x / 2), self.scale * (0.5 + y - self.size_y / 2), 0)
        ) for y in range(self.size_y) for x in range(self.size_x)
    ]

    edges = []
    
    faces = [
        [
            x + self.size_x * y, 
            x + 1 + self.size_x * y,
            x + 1 + self.size_x * (y + 1),
            x + self.size_x * (y + 1)
        ] for y in range(self.size_y - 1) for x in range(self.size_x - 1)
    ]
    
    
    # add water
    mesh = bpy.data.meshes.new(name="Water")
    mesh.from_pydata(verts, edges, faces)
    
    # useful for development when the mesh may be invalid.
    # mesh.validate(verbose=True)
    object_data_add(context, mesh, operator=self)
    
    # TODO takto sa to nerobi
    bpy.context.active_object["size_x"] = self.size_x
    bpy.context.active_object["size_y"] = self.size_y
    bpy.context.active_object["is_water"] = True
    bpy.context.active_object["heightdata"] = [[0.0 for x in range(self.size_x)] for y in range(self.size_y)]
    bpy.context.active_object["previous_heightdata"] = [[0.0 for x in range(self.size_x)] for y in range(self.size_y)]
    bpy.context.active_object["sediment"] = [[0.0 for x in range(self.size_x)] for y in range(self.size_y)]
    bpy.context.active_object["flow"] = [[[0.0, 0.0, 0.0, 0.0] for x in range(self.size_x)] for y in range(self.size_y)]
    bpy.context.active_object["regolith"] = [[0.0 for x in range(self.size_x)] for y in range(self.size_y)]

    
    # z-fighting-fighter
    bpy.context.active_object.location.z -= 0.005  # good enough
    
    # materials
    bpy.context.active_object.data.materials.append(bpy.data.materials.get("water"))
    bpy.ops.object.shade_smooth()

    water_object = bpy.context.active_object    

    
    # add terrain
    # clone arrays, just to be sure
    verts = verts[:]
    edges = []
    faces = faces[:]
    
    mesh = bpy.data.meshes.new(name="Terrain")
    mesh.from_pydata(verts, edges, faces)
    
    # useful for development when the mesh may be invalid.
    # mesh.validate(verbose=True)
    object_data_add(context, mesh, operator=self)
    
    # TODO takto sa to nerobi
    bpy.context.active_object["size_x"] = self.size_x
    bpy.context.active_object["size_y"] = self.size_y
    bpy.context.active_object["is_terrain"] = True
    bpy.context.active_object["heightdata"] = [[0.0 for x in range(self.size_x)] for y in range(self.size_y)]
    
    # materials
    bpy.context.active_object.data.materials.append(bpy.data.materials.get("ground"))
    
    terrain_object = bpy.context.active_object
    
    
    # parent them
    water_object.select_set(True)
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    
    # set terrain as active object
    bpy.context.view_layer.objects.active = terrain_object


class AddTerrainObject(bpy.types.Operator, AddObjectHelper):
    """Create a new Terrain"""
    bl_idname = "mesh.add_terrain"
    bl_label = "Add Terrain"
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatProperty(name="Scale", default=1.0, min=0.0)
    size_x: bpy.props.IntProperty(name="Size X", default=128, min=8)
    size_y: bpy.props.IntProperty(name="Size Y", default=128, min=8)

    def execute(self, context):

        add_object(self, context)

        return {'FINISHED'}
    
    
class InitTerrainObject(bpy.types.Operator, AddObjectHelper):
    """Modify a Terrain object"""
    bl_idname = "mesh.modify_terrain"
    bl_label = "Modify Terrain"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "is_terrain" in context.active_object


    # add noise
    def execute(self, context):
        size_x = context.active_object["size_x"]
        size_y = context.active_object["size_y"]
        heightmap = context.active_object["heightdata"]
        
        def get_noise(x, y):
            sample_scale = 0.025
            position = (sample_scale * x, sample_scale * y, 0)
            fractal_dimension = 1.0
            lacunarity = 3
            octaves = 3
            offset = 0.0
            
            return noise.hetero_terrain(
                position,
                fractal_dimension, 
                lacunarity, 
                octaves,
                offset
            )

        for x in range(size_x):
            for y in range(size_y):
                amplitude = 20.0
                coef_x = x / size_x
                coef_y = y / size_y
                heightmap[x][y] = amplitude * (
                        coef_y * (
                            coef_x * get_noise(x, y) 
                            + (1 - coef_x) * get_noise(x + size_x, y)
                        ) 
                        + (1 - coef_y) * ( 
                            coef_x * get_noise(x, y + size_y)
                            + (1 - coef_x) * get_noise(x + size_x, y + size_y)
                        )
                    )
                ####!
                #heightmap[x][y] = 0.2*x
                
                
        set_heightmap(context, size_x, size_y, heightmap)
        heightmap = context.active_object["heightdata"]
        # update water heights as well
        set_water_heightmap(context, size_x, size_y, [[0.2 for x in range(size_x)] for y in range(size_y)], heightmap)
        
        return {'FINISHED'}
    
    
    
class ErodeTerrainObject(bpy.types.Operator, AddObjectHelper):
    """Erode a Terrain object"""
    bl_idname = "mesh.erode_terrain"
    bl_label = "Erode Terrain"
    bl_options = {'REGISTER', 'UNDO'}
    
    steps: bpy.props.IntProperty(name="Steps", default=1, min=1)
    
    # cyclic array
    class CyclicArray(object):
        def __init__(self, original_array):
            self.size_x = len(original_array)
            self.size_y = len(original_array[0])
            self.arr = original_array
        
        def get(self, x, y):
            x %= self.size_x 
            y %= self.size_y
            return self.arr[x, y]


    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "is_terrain" in context.active_object


    def execute(self, context):
        size_x = context.active_object["size_x"]
        size_y = context.active_object["size_y"]
        
        # (dir x, dir y, index)
        neighbors = [(1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3)]
        length = 1.0
        area = 1.0
        g = 9.81
        c = 0.1
        delta_t = 0.1 
        delta_x = [ 1, 0, -1, 0 ];
        delta_y = [ 0, 1, 0, -1 ];
        
        self.prepare_everything(context)
        

        size_x = self.size_x
        size_y = self.size_y
       
        #water = add_rain(water)
        
        self.heightmap, self.water, self.previous_water, self.sediment, self.flow, self.regolith = erode(size_x, size_y, self.heightmap, self.water, self.previous_water, self.sediment, self.flow, self.regolith)

        """ 
        # whatever
        print("old")
        #TEMPORARY
        steps = 10
        
        for step in range(steps):
            #heightmap2 = [[0 for x in range(size_x)] for y in range(size_y)]
            water2 = [[0 for x in range(size_x)] for y in range(size_y)]
            
            for x in range(size_x):
                for y in range(size_y):
                    if self.water[x][y] > 0:
                        ######################################################
                        # MOVE WATER
                        ######################################################
                        
                        height_here = self.heightmap[x][y] + self.water[x][y]
                        acceleration = [0.0, 0.0, 0.0, 0.0]
                        flow_sum = 0.0
                        
                        # calculate the height differences with neighbors and flow
                        for i in range(4):
                            if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                                height_neighbor = self.heightmap[x + delta_x[i]][y + delta_y[i]] + self.water[x + delta_x[i]][ y + delta_y[i]]
                                delta_height = height_here - height_neighbor
                                acceleration[i] = g * delta_height / length
                                self.flow[x][y][i] = max(0.0, self.flow[x][y][i] + (delta_t * acceleration[i] * length * length))
                                flow_sum += delta_t * self.flow[x][y][i]
                                
                        # scale flow
                        scaling = 1.0
                        if flow_sum > length * length * self.water[x][y]:
                            scaling = length * length * self.water[x][y] / flow_sum
                            flow_sum = length * length * self.water[x][y]
                            
                        # add to neighbors
                        for i in range(4):
                            if (0 <= x + delta_x[i] < size_x) and (0 <= y + delta_y[i] < size_y):
                                self.flow[x][y][i] *= scaling
                                water2[x + delta_x[i]][y + delta_y[i]] += self.flow[x][y][i] * delta_t / (length * length)
                                
                        water2[x][y] += self.water[x][y] - (flow_sum / (length*length))
                        
                        ######################################################
                        # FORCE EROSION
                        ######################################################
                        average_height = 0.5 * (self.water[x][y] + self.previous_water[x][y])

                        # this max function doesn't really have a physical basis, but it makes sure that small amounts of water don't erode ridiculous amounts
                        average_height = max(average_height, 0.2)

                        flow_velocity = self.get_water_velocity_vector(x, y)

                        # steepness and convexness
                        local_shape_factor = 1.0 * (self.get_steepness(x, y)[2] + 0.25 * self.get_convexness(x, y)[2])
                        sediment_capacity = (flow_velocity[2] / (length * average_height)) * c * local_shape_factor
                        
                        if self.sediment[x][y] <= sediment_capacity:
                            # erode
                            amount = 0.5 * (sediment_capacity - self.sediment[x][y])
                            self.heightmap[x][y] -= amount
                            self.sediment[x][y] += amount
                        else:
                            # deposit
                            amount = 0.25 * (self.sediment[x][y] - sediment_capacity)
                            self.heightmap[x][y] += amount
                            self.sediment[x][y] -= amount
                            
                            
                    else:
                        water2[x][y] = self.water[x][y]
                                
                    self.previous_water[x][y] = self.water[x][y]
                    
                    if 1 == 0:
                    #if water[x][y] != 0:
                        # move water
                        out_volume = [0, 0, 0, 0]
                        
                        for (dx, dy, dir) in neighbors:  
                            if not (0 <= x + dx < size_x) or not (0 <= y + dy < size_y):
                                 flow[x][y][dir] = 0
                                 out_volume[dir] = 0
                            else:
                                #delta_height = (heightmap[x][y] + water[x][y]) - (heightmap[(x + dx) % size_x][(y + dy) % size_y] + water[(x + dx) % size_x][(y + dy) % size_y])    
                                delta_height = -(heightmap[x][y] + water[x][y]) + (heightmap[x + dx][y + dy] + water[x + dx][y + dy])    
                                
                                acceleration = g * delta_height / length
                                
                                flow[x][y][dir] = max(0, flow[x][y][dir] + delta_t * area * acceleration)
                                
                                out_volume[dir] = max(0, delta_t * flow[x][y][dir])
                            
                        out_volume_sum = sum(out_volume)
                        
                        # scale water
                        column_water = length * length * water[x][y]
                        
                        if out_volume_sum > column_water and out_volume_sum > 0:
                            for i in range(4):
                                out_volume[i] *= column_water / out_volume_sum
                            out_volume_sum = column_water
                        
                        # move water
                        water2[x][y] -= out_volume_sum      
                        print(water2[x][y]) 
                        
                        for (dx, dy, dir) in neighbors: 
                            if (0 <= x + dx < size_x) and (0 <= y + dy < size_y):
                                water2[(x + dx) % size_x][(y + dy) % size_y] += out_volume[dir]
                                
                        if water2[x][y] < 0:
                            water2[x][y] = 0.0
        
            # update terrain and water heights
            self.water = water2
            #heightmap = heightmap2 """
                        
        # set terrain and water heights
        self.save_everything()
        
        return {'FINISHED'}
    
    
    def prepare_everything(self, context):
        self.context = context
        self.size_x = context.active_object["size_x"]
        self.size_y = context.active_object["size_y"]
        self.heightmap = get_heightmap(context)
        self.water = get_water_heightmap(context)
        self.flow = context.active_object.children[0]["flow"]
        self.previous_water = context.active_object.children[0]["previous_heightdata"]
        self.sediment = context.active_object.children[0]["sediment"]
        self.regolith = context.active_object.children[0]["regolith"]
        
        
    def save_everything(self):
        set_heightmap(self.context, self.size_x, self.size_y, self.heightmap)
        self.heightmap = self.context.active_object["heightdata"]
        set_water_heightmap(self.context, self.size_x, self.size_y, self.water, self.heightmap)
        self.context.active_object.children[0]["flow"] = self.flow
        self.context.active_object.children[0]["sediment"] = self.sediment
        self.context.active_object.children[0]["regolith"] = self.regolith


    # [x, y, magnitude]
    def get_water_velocity_vector(self, x, y):
        left = self.flow[x - 1][y][0] - self.flow[x][y][2] if x > 1 else 0.0 
        right = self.flow[x][y][0] - self.flow[x + 1][y][2] if x < self.size_x - 1 else 0.0 
        bottom = self.flow[x][y - 1][1] - self.flow[x][y][3] if y > 1 else 0.0 
        top = self.flow[x][y][1] - self.flow[x][y + 1][3] if y < self.size_y - 1 else 0.0 
        
        return [0.5 * (left + right), 0.5 * (top + bottom), math.sqrt(0.5 * ((left + right)**2 + (top + bottom)**2))]


    # I decided to code it in the same way normals are computed
    # so the average steepness, or, the first derivative
    def get_steepness(self, x, y):
        delta_x = [ 1, 0, -1, 0 ];
        delta_y = [ 0, 1, 0, -1 ];
        
        # x, y, magnitude
        steepness = [0.0, 0.0, 0.0]
        
        for i in range(4):        
            if (0 <= x + delta_x[i] < self.size_x) and (0 <= y + delta_y[i] < self.size_y):
                delta_h = self.heightmap[x][y] - self.heightmap[x + delta_x[i]][y + delta_y[i]]
                steepness[0] += delta_x[i] * delta_h
                steepness[1] += delta_y[i] * delta_h
        steepness[2] = math.sqrt(steepness[0]**2 + steepness[1]**2)
        return steepness
    
    
    # similar to a second derivative
    # positive = bump
    # negative = hole
    def get_convexness(self, x, y):
        delta_x = [ 1, 0, -1, 0 ];
        delta_y = [ 0, 1, 0, -1 ];
        
        # x, y, magnitude
        convexness = [0.0, 0.0, 0.0]
        
        for i in range(4):        
            if (0 <= x + delta_x[i] < self.size_x) and (0 <= y + delta_y[i] < self.size_y):
                delta_h = self.heightmap[x][y] - self.heightmap[x + delta_x[i]][y + delta_y[i]]
                convexness[0] += abs(delta_x[i]) * delta_h
                convexness[1] += abs(delta_y[i]) * delta_h
        convexness[2] = math.sqrt(convexness[0]**2 + convexness[1]**2)
        return convexness
        

# returns a heightmap from the z coords of vertices... which might not be correct.
# replace with a raycast later
def get_heightmap(context):
    size_x = context.active_object["size_x"]
    size_y = context.active_object["size_y"]
    
    heightmap = [[0.0 for x in range(size_x)] for y in range(size_y)]
    for x in range(size_x):
        for y in range(size_y):
            heightmap[x][y] = context.active_object.data.vertices[y + size_y * x].co.z
    return heightmap
    #return context.active_object["heightdata"]


# replace with... something less stupid
def get_water_heightmap(context):
    size_x = context.active_object["size_x"]
    size_y = context.active_object["size_y"]

    heightmap = get_heightmap(context)
    water_heightmap = [[0.0 for x in range(context.active_object["size_x"])] for y in range(context.active_object["size_y"])]
    for x in range(size_x):
        for y in range(size_y):
            water_heightmap[x][y] = context.active_object.children[0].data.vertices[y + size_y * x].co.z - heightmap[x][y]
    return water_heightmap
    #return context.active_object.children[0]["heightdata"]


# words can't say how dumb this is... but code can
def set_water_heightmap(context, size_x, size_y, water, heightmap):
    for x in range(size_x):
        for y in range(size_y):
            context.active_object.children[0].data.vertices[y + size_y * x].co.z = water[x][y] + heightmap[x][y]
    context.active_object.children[0]["heightdata"] = water    
        

def set_heightmap(context, size_x, size_y, heightmap):
    for x in range(size_x):
        for y in range(size_y):
            context.active_object.data.vertices[y + size_y * x].co.z = heightmap[x][y]
    context.active_object["heightdata"] = heightmap


def add_rain(water):
    rain = 0.15
    
    for x in range(len(water)):
        for y in range(len(water[0])):
            water[x][y] += rain
    
    return water


# UI

class TerrainSidePanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    
    bl_category = "Terrain"
    bl_label = "Terrain settings"
    
    def draw(self, context):
        row = self.layout.row()
        row.operator("mesh.add_terrain", text="Add Terrain")
        row = self.layout.row()
        row.operator("mesh.modify_terrain", text="cesta tretej triedy")
        row = self.layout.row()
        row.operator("mesh.erode_terrain", text="zeroduj")



# Registration

def add_object_button(self, context):
    self.layout.operator(
        AddTerrainObject.bl_idname,
        text="Add Terrain",
        icon='RNDCURVE')  # TODO, custom icon


# This allows you to right click on a button and link to documentation
def add_object_manual_map():
    url_manual_prefix = "https://docs.blender.org/manual/en/latest/"
    url_manual_mapping = (
        ("bpy.ops.mesh.add_object", "scene_layout/object/types.html"),
    )
    return url_manual_prefix, url_manual_mapping


def register():
    bpy.utils.register_class(AddTerrainObject)
    bpy.utils.register_class(InitTerrainObject)
    bpy.utils.register_class(ErodeTerrainObject)
    bpy.utils.register_manual_map(add_object_manual_map)
    bpy.types.VIEW3D_MT_mesh_add.append(add_object_button)
    bpy.utils.register_class(TerrainSidePanel)
    print("version = 20")


def unregister():
    bpy.utils.unregister_class(ErodeTerrainObject)
    bpy.utils.unregister_class(TerrainSidePanel)
    bpy.utils.unregister_class(AddTerrainObject)
    bpy.utils.unregister_class(InitTerrainObject)
    bpy.utils.unregister_manual_map(add_object_manual_map)
    bpy.types.VIEW3D_MT_mesh_add.remove(add_object_button)


if __name__ == "__main__":
    register()

