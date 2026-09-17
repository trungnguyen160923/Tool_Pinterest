"""Headless Blender scene used by blender_renderer.py. Run only through Blender's Python."""

import argparse
import math
import sys

import bpy
from mathutils import Vector


def arguments():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-preview", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--pose", required=True)
    return parser.parse_args(values)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.materials, bpy.data.meshes, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for item in datablocks:
            if item.users == 0:
                datablocks.remove(item)


def look_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def material(name, color, roughness=0.6):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    node = result.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Roughness"].default_value = roughness
    return result


def fabric_material(image_path):
    result = bpy.data.materials.new("Approved_Print_Fabric")
    result.use_nodes = True
    nodes = result.node_tree.nodes
    links = result.node_tree.links
    output = nodes.get("Material Output")
    bsdf = nodes.get("Principled BSDF")
    image = bpy.data.images.load(image_path, check_existing=True)
    image.colorspace_settings.name = "sRGB"
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "REPEAT"
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 165.0
    noise.inputs["Detail"].default_value = 2.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.16
    bump.inputs["Distance"].default_value = 0.035
    bsdf.inputs["Roughness"].default_value = 0.78
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return result


def rounded_box(name, location, scale, mat, bevel=0.12):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    modifier = obj.modifiers.new("Soft edges", "BEVEL")
    modifier.width = bevel
    modifier.segments = 4
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.materials.append(mat)
    return obj


def add_room():
    wall = material("Warm plaster", (0.72, 0.67, 0.58), 0.9)
    floor = material("Oak floor", (0.24, 0.12, 0.055), 0.62)
    rounded_box("Floor", (0, 0, -0.18), (7.0, 6.5, 0.18), floor, 0.03)
    rounded_box("Back wall", (0, 3.2, 3.2), (7.0, 0.16, 3.6), wall, 0.03)


def add_sofa():
    upholstery = material("Linen sofa", (0.57, 0.48, 0.38), 0.88)
    rounded_box("Sofa base", (0, 0.85, 0.76), (3.45, 1.08, 0.45), upholstery, 0.18)
    rounded_box("Sofa back", (0, 1.74, 2.15), (3.45, 0.27, 1.08), upholstery, 0.18)
    rounded_box("Left arm", (-3.15, 0.77, 1.48), (0.30, 1.08, 0.88), upholstery, 0.18)
    rounded_box("Right arm", (3.15, 0.77, 1.48), (0.30, 1.08, 0.88), upholstery, 0.18)
    cushion = material("Cushion", (0.68, 0.59, 0.48), 0.9)
    for x in (-1.65, 0, 1.65):
        rounded_box("Seat cushion", (x, 0.52, 1.28), (0.78, 0.68, 0.17), cushion, 0.12)


def add_bed():
    linen = material("Bed linen", (0.68, 0.62, 0.55), 0.92)
    wood = material("Bed wood", (0.18, 0.08, 0.035), 0.58)
    rounded_box("Bed frame", (0, 0.8, 0.65), (3.35, 2.35, 0.30), wood, 0.12)
    rounded_box("Mattress", (0, 0.8, 1.10), (3.25, 2.25, 0.28), linen, 0.14)
    rounded_box("Headboard", (0, 2.95, 2.20), (3.35, 0.17, 1.38), wood, 0.10)


def add_bench():
    upholstery = material("Bench linen", (0.57, 0.48, 0.38), 0.9)
    wood = material("Bench wood", (0.16, 0.07, 0.03), 0.65)
    rounded_box("Bench", (0, 0.5, 0.92), (3.1, 0.86, 0.30), upholstery, 0.16)
    for x in (-2.65, 2.65):
        rounded_box("Bench leg", (x, 0.5, 0.34), (0.12, 0.12, 0.52), wood, 0.04)


def cloth_surface(pose, fabric):
    columns, rows = 72, 84
    vertices, uvs = [], []
    for row in range(rows + 1):
        v = row / rows
        for column in range(columns + 1):
            u = column / columns
            fold = math.sin(u * math.pi * 5.0 + v * 1.7) * math.sin(v * math.pi * 3.0) * 0.075
            ripple = math.sin(u * math.pi * 10.0 + v * 5.0) * 0.018
            if pose == "bed_foot_throw":
                x = -3.05 + u * 6.10
                y = -1.35 + v * 1.95
                z = 1.42 + fold + ripple
                if v < 0.16:
                    z -= (0.16 - v) * 4.7
                    y -= (0.16 - v) * 0.45
            elif pose == "bench_throw":
                x = -2.75 + u * 5.5
                y = 0.20 + v * 0.62
                z = 1.28 + fold
                if v < 0.30:
                    z -= (0.30 - v) * 3.2
                    y -= (0.30 - v) * 0.75
            else:
                x = -2.85 + u * 3.55
                if v < 0.34:
                    t = v / 0.34
                    y, z = 1.62 - t * 0.92, 2.85 - t * 1.50
                elif v < 0.64:
                    t = (v - 0.34) / 0.30
                    y, z = 0.70 - t * 1.12, 1.35 + fold
                else:
                    t = (v - 0.64) / 0.36
                    y, z = -0.42 - t * 0.27, 1.34 - t * 1.16
                z += fold + ripple
            vertices.append((x, y, z))
            uvs.append((u, v))
    faces = []
    for row in range(rows):
        for column in range(columns):
            start = row * (columns + 1) + column
            faces.append((start, start + 1, start + columns + 2, start + columns + 1))
    mesh = bpy.data.meshes.new("Blanket cloth mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.uv_layers.new(name="Print UV")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            mesh.uv_layers.active.data[loop_index].uv = uvs[mesh.loops[loop_index].vertex_index]
    mesh.update()
    obj = bpy.data.objects.new("UV mapped blanket", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(fabric)
    solidify = obj.modifiers.new("Blanket thickness", "SOLIDIFY")
    solidify.thickness = 0.045
    subdivision = obj.modifiers.new("Smooth cloth", "SUBSURF")
    subdivision.levels = 1
    subdivision.render_levels = 1
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    return obj


def add_camera_and_lights(pose):
    bpy.ops.object.camera_add(location=(7.8, -9.2, 5.4))
    camera = bpy.context.object
    look_at(camera, (-0.4, 0.55, 1.45))
    bpy.context.scene.camera = camera
    camera.data.lens = 52
    key_data = bpy.data.lights.new("Window key", "AREA")
    key_data.energy = 1150
    key_data.shape = "DISK"
    key_data.size = 5.0
    key = bpy.data.objects.new("Window key", key_data)
    bpy.context.collection.objects.link(key)
    key.location = (-4.5, -3.0, 6.3)
    look_at(key, (-0.5, 0.6, 1.2))
    fill_data = bpy.data.lights.new("Soft fill", "AREA")
    fill_data.energy = 420
    fill_data.size = 4.0
    fill = bpy.data.objects.new("Soft fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (4.2, -1.5, 3.8)
    look_at(fill, (0.0, 0.7, 1.3))
    scene = bpy.context.scene
    # Blender's Python enum is version-dependent even when the UI brands it
    # Eevee Next. Prefer the portable build's supported identifier.
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.045, 0.035, 0.025)
    scene.view_settings.look = "AgX - Medium High Contrast"


def main():
    args = arguments()
    clear_scene()
    add_room()
    if args.pose == "bed_foot_throw":
        add_bed()
    elif args.pose == "bench_throw":
        add_bench()
    else:
        add_sofa()
    cloth_surface(args.pose, fabric_material(args.print_preview))
    add_camera_and_lights(args.pose)
    bpy.context.scene.render.filepath = args.output
    bpy.ops.render.render(write_still=True)


main()
