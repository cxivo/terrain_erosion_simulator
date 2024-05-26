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


import json
import bpy
import math
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector, noise
from .eroder import erode


def add_object(self, context):
    self.scale = context.scene.Terrain.scale
    self.size_x = context.scene.Terrain.size[0]
    self.size_y = context.scene.Terrain.size[1]

    # creates a plane
    verts = [
        Vector(
            (self.scale * (0.5 + y - self.size_y / 2), self.scale * (0.5 + x - self.size_x / 2), 0)
        ) for x in range(self.size_x) for y in range(self.size_y)
    ]

    edges = []
    
    faces = [
        [
            y + self.size_y * x, 
            y + 1 + self.size_y * x,
            y + 1 + self.size_y * (x + 1),
            y + self.size_y * (x + 1)
        ] for x in range(self.size_x - 1) for y in range(self.size_y - 1)
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
    bpy.context.active_object["heightdata"] = [[0.0 for y in range(self.size_y)] for x in range(self.size_x)]
    bpy.context.active_object["previous_heightdata"] = [[0.0 for y in range(self.size_y)] for x in range(self.size_x)]
    bpy.context.active_object["sediment"] = [[0.0, 0.0, 0.0, 0.0, 0.0] for y in range(self.size_y) for x in range(self.size_x)]
    bpy.context.active_object["flow"] = [[[0.0, 0.0, 0.0, 0.0] for y in range(self.size_y)] for x in range(self.size_x)]
    bpy.context.active_object["regolith"] = [[0.0, 0.0] for y in range(self.size_y) for x in range(self.size_x)]
    bpy.context.active_object["velocity_x"] = [[0.0 for y in range(self.size_y)] for x in range(self.size_x)]
    bpy.context.active_object["velocity_y"] = [[0.0 for y in range(self.size_y)] for x in range(self.size_x)]


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
    bpy.context.active_object["heightdata"] = [[0.0 for y in range(self.size_y)] for x in range(self.size_x)]
    bpy.context.active_object["topsoil"] = [[0.0, 0.0, 0.0, 0.0, 0.0] for y in range(self.size_y) for x in range(self.size_x)]
    bpy.context.active_object["subsoil"] = [[0.0, 0.0, 0.0, 0.0, 0.0] for y in range(self.size_y) for x in range(self.size_x)]
    bpy.context.active_object["bedrock"] = [[1000000.0] for y in range(self.size_y) for x in range(self.size_x)]
    bpy.context.active_object["wet"] = [[0.0 for y in range(self.size_y)] for x in range(self.size_x)]
    bpy.context.active_object["bedrock_types"] = [0]
    bpy.context.active_object["vegetation"] = [[0.0 for y in range(self.size_y)] for x in range(self.size_x)]

    
    # materials
    bpy.context.active_object.data.materials.append(bpy.data.materials.get("ground"))
    
    terrain_object = bpy.context.active_object
    
    
    # parent them
    water_object.select_set(True)
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    
    # set terrain as active object
    bpy.context.view_layer.objects.active = terrain_object

    print("New terrain, size_x=" + str(self.size_x) + ", size_y=" + str(self.size_y))


class AddTerrainObject(bpy.types.Operator, AddObjectHelper):
    """Create a new Terrain"""
    bl_idname = "mesh.add_terrain"
    bl_label = "Add Terrain"
    bl_options = {'REGISTER', 'UNDO'}

    """ scale: bpy.props.FloatProperty(name="Scale", default=1.0, min=0.0)
    size_x: bpy.props.IntProperty(name="Size X", default=128, min=8)
    size_y: bpy.props.IntProperty(name="Size Y", default=128, min=8)
    """

    def execute(self, context):

        add_object(self, context)

        return {'FINISHED'}
    
    
class InitTerrainObject(bpy.types.Operator):
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
        topsoil = context.active_object["topsoil"]
        subsoil = context.active_object["subsoil"]
        bedrock = context.active_object["bedrock"]
        bedrock_types = context.active_object["bedrock_types"]

        print("heightdata size_x=" + str(len(heightmap)) + ", size_y=" + str(len(heightmap[0])))

        
        def get_noise(x, y, seed):
            sample_scale = context.scene.Terrain.sample_scale / 1000.0
            position = (sample_scale * (x + seed * size_x), sample_scale * (y + seed * size_y), 0)
            fractal_dimension = context.scene.Terrain.fractal_dimension
            lacunarity = context.scene.Terrain.lacunarity
            octaves = context.scene.Terrain.octaves
            offset = context.scene.Terrain.offset
            
            return noise.hetero_terrain(
                position,
                fractal_dimension, 
                lacunarity, 
                octaves,
                offset,
                noise_basis=context.scene.Terrain.noise_type
            )
        
        def get_wrapping_noise(x, y, seed):
            coef_x = x / size_x
            coef_y = y / size_y
            return 1.0 * (
                    coef_y * (
                        coef_x * get_noise(x, y, seed) 
                        + (1 - coef_x) * get_noise(x + size_x, y, seed)
                    ) 
                    + (1 - coef_y) * ( 
                        coef_x * get_noise(x, y + size_y, seed)
                        + (1 - coef_x) * get_noise(x + size_x, y + size_y, seed)
                    )
                )

        # define heights
        amplitude = context.scene.Terrain.amplitude
        
        for x in range(size_x):
            for y in range(size_y):
                heightmap[x][y] = 2.0 + amplitude + amplitude * get_wrapping_noise(x, y, 0)
                
        # define soil layers
        bedrock_types = [0, 1, 2]
        for x in range(size_x):
            for y in range(size_y):
                """ if x < size_x / 4:
                    topsoil[y + size_y * x] = [0.25, 0.1, 0.0, 0.0, 0.9] # 1.25
                    subsoil[y + size_y * x] = [0.25, 0.5, 0.25, 0.0, 1.25] # 2.25
                elif x < size_x / 2:
                    topsoil[y + size_y * x] = [0.0, 0.0, 0.0, 0.0, 1.25] # 1.25
                    subsoil[y + size_y * x] = [0.0, 0.0, 0.0, 0.0, 2.25] # 2.25
                elif x < 3* size_x / 4:
                    topsoil[y + size_y * x] = [1.25, 0.0, 0.0, 0.0, 0] # 1.25
                    subsoil[y + size_y * x] = [0.0, 0.0, 2.25, 0.0, 0] # 2.25
                else:
                    topsoil[y + size_y * x] = [0.0, 0.0, 0.0, 0.0, 0.0] # 1.25
                    subsoil[y + size_y * x] = [0.0, 0.0, 0.0, 0.0, 0.0] # 2.25 """

                topsoil[y + size_y * x] = [0.25, 0.1, 0.0, 0.0, 0.9] # 1.25
                subsoil[y + size_y * x] = [0.25, 0.5, 0.25, 0.0, 1.25] # 2.25
                bedrock[y + size_y * x] = [3.0, 10.0 ,1000000.0] # rest, formerly heightmap[x][y] - 1.25 - 2.25

        # temporary particle system
        if False:
            # vertex group
            vertex_group = context.active_object.vertex_groups.new(name = 'verteks_grup')

            # particle system
            bpy.ops.object.particle_system_add()
            particle_system_name = bpy.data.particles[-1].name
            # set it to hair -> object
            bpy.data.particles[particle_system_name].type = 'HAIR'
            bpy.data.particles[particle_system_name].render_type = 'OBJECT'
            bpy.data.particles[particle_system_name].child_type = 'INTERPOLATED'
            # set it to the object Boulder
            bpy.data.particles[particle_system_name].instance_object = bpy.data.objects["boulder"]


            for x in range(size_x):
                for y in range(size_y):
                    vertex_group.add([y + size_y * x], min(max(heightmap[x][y] - 20.0, 0.0), 1.0), 'REPLACE')

            # assign vertex group
            bpy.context.active_object.particle_systems[0].vertex_group_density = 'verteks_grup' 


        # create a texture image
        # try deleting the old one
        is_old_one_good = False
        try:
            """ old = bpy.data.images['texture_type']
            if old.size[0] == size_x and old.size[1] == size_y:
                is_old_one_good = True
            else: """
            bpy.data.images.remove(bpy.data.images['texture_type'])
            bpy.data.images.remove(bpy.data.images['texture_velocity'])
            bpy.data.images.remove(bpy.data.images['texture_organic'])
            bpy.data.images.remove(bpy.data.images['texture_rock0'])
            bpy.data.images.remove(bpy.data.images['texture_rock1'])
            bpy.data.images.remove(bpy.data.images['texture_rock2'])
            bpy.data.images.remove(bpy.data.images['texture_rock3'])
            bpy.data.images.remove(bpy.data.images['texture_bedrock0'])
            bpy.data.images.remove(bpy.data.images['texture_bedrock1'])
            bpy.data.images.remove(bpy.data.images['texture_bedrock2'])
            bpy.data.images.remove(bpy.data.images['texture_water'])
            bpy.data.images.remove(bpy.data.images['texture_vegetation'])
        except:
            pass

        if not is_old_one_good:
            bpy.data.images.new('texture_type', size_x, size_y)
            bpy.data.images.new('texture_velocity', size_x, size_y)
            bpy.data.images.new('texture_organic', size_x, size_y)
            bpy.data.images.new('texture_rock0', size_x, size_y)
            bpy.data.images.new('texture_rock1', size_x, size_y)
            bpy.data.images.new('texture_rock2', size_x, size_y)
            bpy.data.images.new('texture_rock3', size_x, size_y)
            bpy.data.images.new('texture_bedrock0', size_x, size_y)
            bpy.data.images.new('texture_bedrock1', size_x, size_y)
            bpy.data.images.new('texture_bedrock2', size_x, size_y)
            bpy.data.images.new('texture_water', size_x, size_y)
            bpy.data.images.new('texture_vegetation', size_x, size_y)
        

        #topsoil_texture = bpy.data.images['topsoil_texture']

        

        context.active_object["topsoil"] = topsoil
        context.active_object["subsoil"] = subsoil
        context.active_object["bedrock"] = bedrock
        context.active_object["bedrock_types"] = bedrock_types

        set_heightmap(context, size_x, size_y, heightmap)
        heightmap = context.active_object["heightdata"]
        # update water heights as well
        set_water_heightmap(context, size_x, size_y, [[0.2 for y in range(size_y)] for x in range(size_x)], heightmap)
        
        return {'FINISHED'}
    
    
    
class ErodeTerrainObject(bpy.types.Operator):
    """Erode a Terrain object"""
    bl_idname = "mesh.erode_terrain"
    bl_label = "Erode Terrain"
    bl_options = {'REGISTER', 'UNDO'}

    
    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "is_terrain" in context.active_object


    def execute(self, context):
        size_x = context.active_object["size_x"]
        size_y = context.active_object["size_y"]
        
        self.prepare_everything(context)
        
        size_x = self.size_x
        size_y = self.size_y

        # props
        steps = context.scene.Terrain.steps
        delta_t = context.scene.Terrain.delta_t
        erosion_constant = context.scene.Terrain.erosion_constant
        inertial_erosion_constant = context.scene.Terrain.inertial_erosion_constant
        max_penetration_depth = context.scene.Terrain.max_penetration_depth
        downhill_creep_constant = context.scene.Terrain.downhill_creep_constant
        vegetation_spread_speed = context.scene.Terrain.vegetation_spread_speed
        vegetation_base = context.scene.Terrain.vegetation_base
        organification_speed = context.scene.Terrain.organification_speed
        soilification_speed = context.scene.Terrain.soilification_speed

        is_river = context.scene.Terrain.is_river
        source_height = context.scene.Terrain.source_height
       

        # the main thing
        self.heightmap, self.water, self.previous_water, self.sediment, self.flow, self.regolith, self.topsoil, self.subsoil, self.bedrock, self.bedrock_types, self.wet, self.vegetation, texture_type, texture_velocity, texture_organic, texture_rock, texture_bedrock, texture_water, texture_vegetation = erode(steps, delta_t, erosion_constant, max_penetration_depth, inertial_erosion_constant, downhill_creep_constant, vegetation_spread_speed, vegetation_base, organification_speed, soilification_speed, is_river, source_height, size_x, size_y, self.heightmap, self.water, self.previous_water, self.sediment, self.flow, self.regolith, self.topsoil, self.subsoil, self.bedrock, self.bedrock_types, self.velocity_x, self.velocity_y, self.wet, self.vegetation)

        bpy.data.images['texture_type'].pixels = texture_type
        bpy.data.images['texture_velocity'].pixels = texture_velocity
        bpy.data.images['texture_organic'].pixels = texture_organic
        bpy.data.images['texture_rock0'].pixels = texture_rock[0]
        bpy.data.images['texture_rock1'].pixels = texture_rock[1]
        bpy.data.images['texture_rock2'].pixels = texture_rock[2]
        bpy.data.images['texture_rock3'].pixels = texture_rock[3]
        bpy.data.images['texture_bedrock0'].pixels = texture_bedrock[0]
        bpy.data.images['texture_bedrock1'].pixels = texture_bedrock[1]
        bpy.data.images['texture_bedrock2'].pixels = texture_bedrock[2]
        bpy.data.images['texture_water'].pixels = texture_water
        bpy.data.images['texture_vegetation'].pixels = texture_vegetation

                        
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
        self.topsoil = context.active_object["topsoil"]
        self.subsoil = context.active_object["subsoil"]
        self.bedrock = context.active_object["bedrock"]
        self.wet = context.active_object["wet"]
        self.vegetation = context.active_object["vegetation"]
        self.bedrock_types = context.active_object["bedrock_types"].to_list()
        self.velocity_x = context.active_object.children[0]["velocity_x"]
        self.velocity_y = context.active_object.children[0]["velocity_y"]
        
        
    def save_everything(self):
        set_heightmap(self.context, self.size_x, self.size_y, self.heightmap)
        self.heightmap = self.context.active_object["heightdata"]
        set_water_heightmap(self.context, self.size_x, self.size_y, self.water, self.heightmap)
        self.context.active_object.children[0]["flow"] = self.flow
        self.context.active_object.children[0]["sediment"] = self.sediment
        self.context.active_object.children[0]["regolith"] = self.regolith
        self.context.active_object["topsoil"] = self.topsoil 
        self.context.active_object["subsoil"] = self.subsoil
        self.context.active_object["bedrock"] = self.bedrock
        self.context.active_object["wet"] = self.wet
        self.context.active_object["vegetation"] = self.vegetation
        self.context.active_object["bedrock_types"] = self.bedrock_types
        self.context.active_object.children[0]["velocity_x"] = self.velocity_x
        self.context.active_object.children[0]["velocity_y"] = self.velocity_y
        

# returns a heightmap from the z coords of vertices... which might not be correct.
# replace with a raycast later
def get_heightmap(context):
    size_x = context.active_object["size_x"]
    size_y = context.active_object["size_y"]
    
    heightmap = [[0.0 for y in range(size_y)] for x in range(size_x)]
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
    water_heightmap = [[0.0 for y in range(context.active_object["size_y"])] for x in range(context.active_object["size_x"])]
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



class DrainWater(bpy.types.Operator):
    """Drain all water"""
    bl_idname = "mesh.drain_water"
    bl_label = "Drain water"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "is_terrain" in context.active_object


    def execute(self, context):
        size_x = context.active_object["size_x"]
        size_y = context.active_object["size_y"]
        set_water_heightmap(context, size_x, size_y, [[-1.0 for y in range(size_y)] for x in range(size_x)], get_heightmap(context))

        return {'FINISHED'}


class SetWaterLevel(bpy.types.Operator):
    """Set all water to level"""
    bl_idname = "mesh.set_water_level"
    bl_label = "Set water level"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and "is_terrain" in context.active_object


    def execute(self, context):
        size_x = context.active_object["size_x"]
        size_y = context.active_object["size_y"]
        old_water = get_water_heightmap(context)
        rain_height = context.scene.Terrain.rain_height

        new_water = [[rain_height + max(old_water[x][y], 0.0) for y in range(size_y)] for x in range(size_x)]
        set_water_heightmap(context, size_x, size_y, new_water, get_heightmap(context))

        return {'FINISHED'}


##################################################
# UI

class TERRAIN_PT_Sidebar(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Terrain"
    bl_label = "Terrain settings"
    
    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="New terrain")

        # Add terrain
        box.prop(context.scene.Terrain, "scale")
        
        box.prop(context.scene.Terrain, "size")

        box.operator("mesh.add_terrain", text="Add Terrain")
        
        # Noise
        box = layout.box()
        box.label(text="Noise")

        box.prop(context.scene.Terrain, "sample_scale")
        box.prop(context.scene.Terrain, "amplitude")
        box.prop(context.scene.Terrain, "fractal_dimension")
        box.prop(context.scene.Terrain, "lacunarity")
        box.prop(context.scene.Terrain, "octaves")
        box.prop(context.scene.Terrain, "offset")
        box.prop(context.scene.Terrain, "noise_type")

        box.operator("mesh.modify_terrain", text="Apply noise")
        
        # Erosion
        box = layout.box()
        box.label(text="Erosion")

        box.prop(context.scene.Terrain, "steps")
        box.prop(context.scene.Terrain, "delta_t")
        
        col = box.column()
        col.prop(context.scene.Terrain, "erosion_constant")
        col.prop(context.scene.Terrain, "inertial_erosion_constant")
        col.prop(context.scene.Terrain, "regolith_constant")
        col.prop(context.scene.Terrain, "max_penetration_depth")
        col.prop(context.scene.Terrain, "downhill_creep_constant")
        col.prop(context.scene.Terrain, "vegetation_spread_speed")
        col.prop(context.scene.Terrain, "vegetation_base")
        col.prop(context.scene.Terrain, "organification_speed")
        col.prop(context.scene.Terrain, "soilification_speed")

        col = box.column()
        col.prop(context.scene.Terrain, "is_river")
        col.prop(context.scene.Terrain, "source_height")

        col = box.column()
        box.operator("mesh.drain_water", text="Drain water")
        col.prop(context.scene.Terrain, "rain_height")
        box.operator("mesh.set_water_level", text="Set water level")

        box.operator("mesh.erode_terrain", text="Erode")


# Property definitions
class TERRAIN_PG_SceneProperties(bpy.types.PropertyGroup):
    scale: bpy.props.FloatProperty(name="Scale", default=1.0, min=0.0, description="What resolution the terrain is")
    size: bpy.props.IntVectorProperty(name="Size", size=2, default=(128, 128), min=8, description="Number of cells of the terrain")

    sample_scale: bpy.props.FloatProperty(name="Sample scale", default=25.0, min=0.0, description="Scale of the noise, divided by 1000")
    amplitude: bpy.props.FloatProperty(name="Amplitude", default=15.0, description="Amplitude of the noise")
    fractal_dimension: bpy.props.FloatProperty(name="Fractal dimension", default=1.0, min=0.0)
    lacunarity: bpy.props.FloatProperty(name="Lacunarity", default=3.0, min=0.0)
    octaves: bpy.props.FloatProperty(name="Octaves", default=3, min=0)
    offset: bpy.props.FloatProperty(name="Offset", default=1.0)

    # identifier, what user sees, description
    noise_types_enum = [('BLENDER', 'BLENDER', "" ), ('PERLIN_ORIGINAL', 'PERLIN_ORIGINAL', ""), ('PERLIN_NEW', 'PERLIN_NEW', ""), ('VORONOI_F1', 'VORONOI_F1', ""), ('VORONOI_F2', 'VORONOI_F2', ""), ('VORONOI_F3', 'VORONOI_F3', ""), ('VORONOI_F4', 'VORONOI_F4', ""), ('VORONOI_F2F1', 'VORONOI_F2F1', ""), ('VORONOI_CRACKLE', 'VORONOI_CRACKLE', ""), ('CELLNOISE', 'CELLNOISE', "")]

    noise_type: bpy.props.EnumProperty(name="Noise type", items=noise_types_enum, default='PERLIN_ORIGINAL', description="Type of noise function")

    steps: bpy.props.IntProperty(name="Steps", default=1, min=1, description="Number of iterations to be performed at once")
    delta_t: bpy.props.FloatProperty(name="delta t", default=0.1, min=0.0000001, max=10, description="Delta time, smaller = slower & finer, bigger = faster & rougher")
    erosion_constant: bpy.props.FloatProperty(name="erosion constant", default=0.25, min=0.0, max=10, description="Multiplier for the force erosion")
    inertial_erosion_constant: bpy.props.FloatProperty(name="inertial erosion constant", default=0.25, min=0.0, max=100, description="Multiplier for the inertial erosion")
    downhill_creep_constant: bpy.props.FloatProperty(name="downhill creep constant", default=0.01, min=0.0, max=10.0, description="How quickly the  downhill creep should act")
    vegetation_spread_speed: bpy.props.FloatProperty(name="vegetation spread speed", default=1, min=0.0, max=10.0, description="How quickly should the vegetation change")
    vegetation_base: bpy.props.FloatProperty(name="vegetation base", default=0.20, min=0.0, max=2.0, description="How sensitive is vegetation to soil content")
    organification_speed: bpy.props.FloatProperty(name="rock breakdown speed", default=0.1, min=0.0, max=100.0, description="How quickly vegetation breaks down rocks")
    soilification_speed: bpy.props.FloatProperty(name="bedrock to organic conversion", default=0.1, min=0.0, max=100.0, description="How quickly vegetation breaks down bedrock into smaller rocks")

    max_penetration_depth: bpy.props.FloatProperty(name="maximum regolith penetration depth", default=0.01, min=0.0, max=2, description="Maximum depth of the regolith layer")

    is_river: bpy.props.BoolProperty(name="River mode", default=False, description="A water source will spawn on one side of the map, all water will be drained from the other side")
    source_height: bpy.props.FloatProperty(name="River source height", default=1.0, min=0.0, max=20.0, description="Height of the river water source column")

    rain_height: bpy.props.FloatProperty(name="Rain height", default=0.1, min=0.0, max=20.0, description="Height of water to be set by the rain")



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
    bpy.utils.register_class(DrainWater)
    bpy.utils.register_class(SetWaterLevel)
    bpy.utils.register_manual_map(add_object_manual_map)
    bpy.types.VIEW3D_MT_mesh_add.append(add_object_button)
    bpy.utils.register_class(TERRAIN_PT_Sidebar)
    bpy.utils.register_class(TERRAIN_PG_SceneProperties)
    bpy.types.Scene.Terrain = bpy.props.PointerProperty(type=TERRAIN_PG_SceneProperties)
    print("version = 20")


def unregister():
    bpy.utils.unregister_class(ErodeTerrainObject)
    bpy.utils.unregister_class(TERRAIN_PG_SceneProperties)
    bpy.utils.unregister_class(TERRAIN_PT_Sidebar)
    bpy.utils.unregister_class(AddTerrainObject)
    bpy.utils.unregister_class(InitTerrainObject)
    bpy.utils.unregister_class(DrainWater)
    bpy.utils.unregister_class(SetWaterLevel)
    bpy.utils.unregister_manual_map(add_object_manual_map)
    bpy.types.VIEW3D_MT_mesh_add.remove(add_object_button)
    del bpy.types.Scene.Terrain


if __name__ == "__main__":
    register()

