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
    bpy.context.active_object["sediment"] = [[0.0, 0.0, 0.0, 0.0, 0.0] for y in range(self.size_y) for x in range(self.size_x)]
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
    bpy.context.active_object["topsoil"] = [[0.0, 0.0, 0.0, 0.0, 0.0] for y in range(self.size_y) for x in range(self.size_x)]
    bpy.context.active_object["subsoil"] = [[0.0, 0.0, 0.0, 0.0, 0.0] for y in range(self.size_y) for x in range(self.size_x)]
    bpy.context.active_object["bedrock"] = [[0, 1000000.0] for y in range(self.size_y) for x in range(self.size_x)]
    
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
        topsoil = context.active_object["topsoil"]
        subsoil = context.active_object["subsoil"]
        bedrock = context.active_object["bedrock"]

        
        def get_noise(x, y, seed):
            sample_scale = 0.025
            position = (sample_scale * (x + seed * size_x), sample_scale * (y + seed * size_y), 0)
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
        for x in range(size_x):
            for y in range(size_y):
                amplitude = 15.0
                heightmap[x][y] = 2.0 + amplitude + amplitude * get_wrapping_noise(x, y, 0)
                
        # define soil layers
        for x in range(size_x):
            for y in range(size_y):
                topsoil[y + size_y * x] = [0.25, 0.1, 0.0, 0.0, 0.9] # 1.25
                subsoil[y + size_y * x] = [0.25, 0.5, 0.25, 0.0, 1.25] # 2.25
                bedrock[y + size_y * x] = [0, 1000000.0] # rest, formerly heightmap[x][y] - 1.25 - 2.25

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
        bpy.data.images.new('topsoil_texture', size_x, size_y)
        #topsoil_texture = bpy.data.images['topsoil_texture']

        

        context.active_object["topsoil"] = topsoil
        context.active_object["subsoil"] = subsoil
        context.active_object["bedrock"] = bedrock

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
        
        self.prepare_everything(context)
        
        size_x = self.size_x
        size_y = self.size_y

        # props
        steps = context.scene.Terrain.steps
        delta_t = context.scene.Terrain.delta_t
        erosion_constant = context.scene.Terrain.erosion_constant
        inertial_erosion_constant = context.scene.Terrain.inertial_erosion_constant
        max_penetration_depth = context.scene.Terrain.max_penetration_depth
        regolith_constant = context.scene.Terrain.regolith_constant

        is_river = context.scene.Terrain.is_river
        source_height = context.scene.Terrain.source_height
       
        #water = add_rain(water)
        # the main thing
        self.heightmap, self.water, self.previous_water, self.sediment, self.flow, self.regolith, self.topsoil, self.subsoil, self.bedrock, topsoil_texture = erode(steps, delta_t, erosion_constant, max_penetration_depth, regolith_constant, inertial_erosion_constant, is_river, source_height, size_x, size_y, self.heightmap, self.water, self.previous_water, self.sediment, self.flow, self.regolith, self.topsoil, self.subsoil, self.bedrock)

        bpy.data.images['topsoil_texture'].pixels = topsoil_texture
                        
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

class TERRAIN_PT_Sidebar(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Terrain"
    bl_label = "Terrain settings"
    
    def draw(self, context):
        layout = self.layout
        
        layout.operator("mesh.add_terrain", text="Add Terrain")
        layout.operator("mesh.modify_terrain", text="cesta tretej triedy")
        
        layout.prop(context.scene.Terrain, "steps")
        layout.prop(context.scene.Terrain, "delta_t")
        
        #row = self.layout.row()
        col = layout.column()
        col.prop(context.scene.Terrain, "erosion_constant")
        col.prop(context.scene.Terrain, "inertial_erosion_constant")
        col.prop(context.scene.Terrain, "regolith_constant")
        col.prop(context.scene.Terrain, "max_penetration_depth")

        col = layout.column()
        col.prop(context.scene.Terrain, "is_river")
        col.prop(context.scene.Terrain, "source_height")

        layout.operator("mesh.erode_terrain", text="zeroduj")


# Property definitions
class TERRAIN_PG_SceneProperties(bpy.types.PropertyGroup):
    steps: bpy.props.IntProperty(name="Steps", default=1, min=1)
    delta_t: bpy.props.FloatProperty(name="delta t", default=0.1, min=0.0000001, max=10)
    erosion_constant: bpy.props.FloatProperty(name="erosion constant", default=0.25, min=0.0, max=10)
    inertial_erosion_constant: bpy.props.FloatProperty(name="inertial erosion constant", default=0.25, min=0.0, max=100)
    regolith_constant: bpy.props.FloatProperty(name="regolith constant", default=0.5, min=0.0, max=10)
    max_penetration_depth: bpy.props.FloatProperty(name="maximum regolith penetration depth", default=0.01, min=0.0, max=2)
    is_river: bpy.props.BoolProperty(name="River mode", default=False)
    source_height: bpy.props.FloatProperty(name="River source height", default=1.0, min=0.0, max=20.0)



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
    bpy.utils.unregister_manual_map(add_object_manual_map)
    bpy.types.VIEW3D_MT_mesh_add.remove(add_object_button)
    del bpy.types.Scene.Terrain


if __name__ == "__main__":
    register()

