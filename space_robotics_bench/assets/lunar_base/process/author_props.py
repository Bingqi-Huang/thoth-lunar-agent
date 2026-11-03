"""Author the small interaction props no downloaded asset covers (CC0, this project).

    blender -b --factory-startup -P process/author_props.py -- <out_dir>

Writes <out_dir>/<name>/<name>.glb plus <out_dir>/<name>/frames.json (named interaction frames in the
prop's own frame: metres, +X = the side the robot works from, Z up, origin at the mount point).

- staging_stand: logistics table, top surface 0.55 m, front (+X) edge free for the arm, back lip,
  two painted placement slots. receiving_stand: the same stand with one slot (a delivery point).
- pdu_service_panel: mounts on a cabinet front face (x = 0 is the face). Red mushroom reset button at
  z 0.78, status lamp (material "StatusLamp", emissive, recoloured at runtime) at z 1.02.
- pdu_service_shelf: bracket shelf beside the panel (top z 0.50, 0.75 deep, 0.72 wide) for spare modules: about 10 cm
  clear of a 0.45 x 0.39 m ORU on every side, since a side-grasped module swings on its handle (research F78).
- pdu_oru_bay: the unit's ORU bay -- a tray of the spares shelf's form (so an installed ORU is picked and placed like a
  spare) with a connector backplate and a black/yellow bay border, so it reads as the unit's own bay, not storage.
- pdu_radiator: the unit's radiator (vertical fin panel) with a roll-up dust shutter: the housing is fixed, the drawn
  shutter `cover_closed` is a visual-only part the task shows or hides (research heldout_design §4: no joint), and a
  cover actuator box carrying a blue mushroom push button. Lunar Roving Vehicle radiators had covers the crew opened
  to let the batteries cool (Creel, DAP-2017).
"""

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

out = Path(sys.argv[sys.argv.index("--") + 1])


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def material(name, color, roughness=0.5, metallic=0.0, emission=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1.0)
    p.inputs["Roughness"].default_value = roughness
    p.inputs["Metallic"].default_value = metallic
    if emission:
        p.inputs["Emission Color"].default_value = (*emission, 1.0)
        p.inputs["Emission Strength"].default_value = 1.5  # same as set_status_lamp(); higher saturates the hue
    return m


def box(name, size, center, mat, bevel=0.004):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    o = bpy.context.object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = o.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        bpy.ops.object.modifier_apply(modifier=mod.name)
    o.data.materials.append(mat)
    return o


def cylinder(name, radius, depth, center, axis, mat, verts=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=center)
    o = bpy.context.object
    o.name = name
    if axis == "X":
        o.rotation_euler = (0.0, math.pi / 2, 0.0)
    elif axis == "Y":
        o.rotation_euler = (math.pi / 2, 0.0, 0.0)
    bpy.ops.object.transform_apply(rotation=True)
    o.data.materials.append(mat)
    return o


def dome(name, radius, center, axis, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=center)
    o = bpy.context.object
    o.name = name
    bm = bmesh.new()
    bm.from_mesh(o.data)
    # keep the half pointing along +axis
    k = 0 if axis == "X" else 2
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co[k] < -1e-6], context="VERTS")
    bm.to_mesh(o.data)
    bm.free()
    o.data.materials.append(mat)
    return o


def export(name, frames):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(d / f"{name}.glb"), export_format="GLB", export_apply=True)
    (d / "frames.json").write_text(json.dumps(frames, indent=1))
    print(f"AUTHORED {name}: {len(bpy.context.scene.objects)} parts", flush=True)


alu = lambda: material("Aluminium", (0.62, 0.63, 0.65), 0.35, 0.9)  # noqa: E731
yellow = lambda: material("SafetyYellow", (0.95, 0.72, 0.05), 0.45)  # noqa: E731

# ---- staging stand (two slots) and receiving stand (one slot): one stand form ----------------------------------------
TOP, D, T = 0.55, 0.60, 0.03  # top height, depth (x), plate thickness


def stand(name, slot_ys, W):
    """A logistics stand: rubber-matted top at TOP on four legs, a yellow front edge the arm works over, a back lip, a
    lower shelf, a painted outline per slot. `slot_ys` are the slot centres across the stand (y); W its width."""
    reset()
    m_alu, m_mat, m_yel, m_paint = alu(), material("RubberMat", (0.09, 0.09, 0.10), 0.9), yellow(), material("SlotPaint", (0.9, 0.9, 0.88), 0.7)
    box("top_plate", (D, W, T), (0, 0, TOP - T / 2), m_mat)
    box("front_edge", (0.03, W, T + 0.004), (D / 2 - 0.015, 0, TOP - T / 2), m_yel, bevel=0.003)
    box("back_lip", (0.03, W, 0.06), (-D / 2 + 0.015, 0, TOP + 0.03), m_alu)
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f"leg_{sx}{sy}", (0.05, 0.05, TOP - T), (sx * (D / 2 - 0.045), sy * (W / 2 - 0.045), (TOP - T) / 2), m_alu)
    for sy in (-1, 1):
        box(f"rail_{sy}", (D - 0.14, 0.04, 0.04), (0, sy * (W / 2 - 0.045), 0.14), m_alu)
    box("shelf", (D - 0.14, W - 0.14, 0.02), (0, 0, 0.16), m_alu)
    slots = {}
    for i, sy in enumerate(slot_ys):
        cx = D / 2 - 0.19  # slot centre 0.19 m behind the front edge (a 0.39 m deep spare stays on the top)
        for j, (sx_, sy_, lx, ly) in enumerate(((0, 0.23, 0.40, 0.012), (0, -0.23, 0.40, 0.012), (0.20, 0, 0.012, 0.46), (-0.20, 0, 0.012, 0.46))):
            box(f"slot{i}_line{j}", (lx, ly, 0.002), (cx + sx_, sy + sy_, TOP + 0.001), m_paint, bevel=0)
        slots[f"slot_{i}"] = dict(type="place", xyz=[cx, sy, TOP], normal=[1, 0, 0])
    export(name, dict(size=[D, W, TOP + 0.06], front_edge_x=D / 2, top_z=TOP, **slots))


stand("staging_stand", (-0.32, 0.32), 1.30)  # 1.30 m: two 0.45 m spares 0.19 m apart (F46)
stand("receiving_stand", (0.0,), 0.70)  # one slot: a delivery point (habitat logistics, depot returns)

# ---- PDU service panel ----------------------------------------------------------------------------
reset()
m_alu, m_yel = alu(), yellow()
m_btn = material("ResetButton", (0.75, 0.04, 0.03), 0.35)
m_lamp = material("StatusLamp", (0.2, 0.9, 0.25), 0.2, emission=(0.1, 1.0, 0.2))
m_dark = material("PanelDark", (0.12, 0.12, 0.13), 0.6)
BTN_Z, LAMP_Z = 0.78, 1.02
box("mount_plate", (0.02, 0.34, 0.52), (0.01, 0.0, 0.86), m_yel)
PLATE_FACE = 0.02  # the mount plate's front face, x
BTN_COLLAR_R, BTN_COLLAR_X, BTN_CAP_R, BTN_CAP_X = 0.036, 0.03, 0.032, 0.06
cylinder("button_collar", BTN_COLLAR_R, 0.02, (BTN_COLLAR_X, 0.0, BTN_Z), "X", m_dark)
cylinder("button_stem", 0.018, 0.02, (0.05, 0.0, BTN_Z), "X", m_dark)
dome("button_cap", BTN_CAP_R, (BTN_CAP_X, 0.0, BTN_Z), "X", m_btn)
cylinder("lamp_base", 0.04, 0.02, (0.03, 0.0, LAMP_Z), "X", m_dark)
dome("status_lamp", 0.034, (0.04, 0.0, LAMP_Z), "X", m_lamp)
export("pdu_service_panel", dict(
    # the button as built, for whoever has to find it: a mushroom on the plate -- a collar and a dome -- heights above the
    # plate's front face; `travel_m` is how far past the cap's tip the tool goes for it to actuate
    press=dict(type="press", xyz=[BTN_CAP_X + BTN_CAP_R, 0.0, BTN_Z], normal=[1, 0, 0],
               interface=dict(kind="mushroom_button", cap_radius_m=BTN_CAP_R, cap_centre_height_m=BTN_CAP_X - PLATE_FACE,
                              collar_radius_m=BTN_COLLAR_R, collar_top_height_m=BTN_COLLAR_X + 0.01 - PLATE_FACE, travel_m=0.01)),
    lamp=dict(type="indicator", xyz=[0.04 + 0.034, 0.0, LAMP_Z], normal=[1, 0, 0], material="StatusLamp"),
))

# ---- PDU service shelf and ORU bay: one tray form ------------------------------------------------------------------
# 0.75 deep, because of what has to fit on it. The spare ORU is checked as a 0.485 m envelope (its 0.392 body widened by
# the 16 deg pivot the bags measured), the side lips take ~28 mm off each end, and the cabinet wall is directly behind it:
# at 0.60 m the item sat 58 mm from that wall and the approach spent it, stalling `place` 5.5 cm out (F85/F86); 0.75 m
# leaves 158 mm behind it, against the 60 mm band a dock is verified to. Depth only -- a tray scaled up all round would
# rise out of the arm's working band. Built at that depth here rather than stretched in processing: a stretched tray
# kept declaring its authored 0.60 as `depth_m`, and the place skill puts the wall it looks for half that behind the point.
SHELF_Z, SHELF_D, SHELF_W = 0.50, 0.75, 0.72


def tray(m_frame):
    """The shelf's tray: plate, back strip, side lips, gussets. The ORU bay is the same tray, so an ORU installed in a bay
    is picked and placed exactly as a spare on a shelf is."""
    box("back_strip", (0.012, SHELF_W, 0.36), (0.006, 0.0, SHELF_Z - 0.16), m_frame)
    box("shelf_plate", (SHELF_D, SHELF_W, 0.02), (SHELF_D / 2, 0.0, SHELF_Z - 0.01), m_frame)
    for sy in (-1, 1):
        box(f"side_lip_{sy}", (SHELF_D, 0.012, 0.03), (SHELF_D / 2, sy * (SHELF_W / 2 - 0.006), SHELF_Z + 0.015), m_frame, bevel=0.002)
        bm = bmesh.new()
        pts = [(0.0, sy * (SHELF_W / 2 - 0.03), SHELF_Z - 0.02), (SHELF_D - 0.03, sy * (SHELF_W / 2 - 0.03), SHELF_Z - 0.02), (0.0, sy * (SHELF_W / 2 - 0.03), SHELF_Z - 0.32)]
        vs = [bm.verts.new(Vector((x, y - 0.006, z))) for x, y, z in pts] + [bm.verts.new(Vector((x, y + 0.006, z))) for x, y, z in pts]
        bmesh.ops.convex_hull(bm, input=vs)
        me = bpy.data.meshes.new(f"gusset_{sy}")
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(f"gusset_{sy}", me)
        bpy.context.scene.collection.objects.link(o)
        o.data.materials.append(m_frame)


# the place point, and what the tray is, for whoever has to set something down on it: how deep and wide the plate is and
# the side lips that stand along its edges. The point sits 0.40 out, not at the plate's centre: at least the item's
# half-envelope from the wall, and it carries the rover's work pose outward with it, which is what stops the deeper tray
# reaching into where the rover parks (stretched 1.42 instead of 1.25, its front edge was 30 mm inside that
# footprint; the scene builder said so).
TRAY_POINT = dict(type="place", xyz=[0.40, 0.0, SHELF_Z], normal=[1, 0, 0],
                  interface=dict(kind="shelf", depth_m=SHELF_D, width_m=SHELF_W, lip_height_m=0.03, lip_thickness_m=0.012))

reset()
tray(alu())
export("pdu_service_shelf", dict(shelf=TRAY_POINT))

reset()
m_frame, m_dark, m_yel = material("BayFrame", (0.20, 0.21, 0.23), 0.5, 0.6), material("PanelDark", (0.12, 0.12, 0.13), 0.6), yellow()
tray(m_frame)
# connector backplate behind the module (x <= 0.05, clear of the ORU's back face at x ~0.1) and a hazard-striped
# border along the plate's front edge, below the lip line, so nothing stands in the tray's working volume
box("connector_plate", (0.02, 0.36, 0.26), (0.022, 0.0, SHELF_Z + 0.13), m_dark)
for j, y in enumerate((-0.12, 0.0, 0.12)):
    box(f"connector_{j}", (0.03, 0.06, 0.05), (0.047, y, SHELF_Z + 0.07), m_dark)
for j in range(9):
    box(f"border_{j}", (0.012, SHELF_W / 9, 0.02), (SHELF_D + 0.006, -SHELF_W / 2 + (j + 0.5) * SHELF_W / 9, SHELF_Z - 0.01),
        m_yel if j % 2 == 0 else m_dark, bevel=0)
# the working surface is the tray, as on the shelf: the connector plate behind the module must not raise the whole prop's
# obstacle box to its own top (the F79/F86 phantom-obstacle mistake)
# `role`: what the site survey says this place is -- the unit's own module bay, not storage
export("pdu_oru_bay", dict(tray=dict(TRAY_POINT, role="equipment_bay"), top_z=SHELF_Z + 0.03))

# ---- PDU radiator with its dust cover and cover actuator ---------------------------------------------------------------
reset()
m_alu, m_dark, m_fin = alu(), material("PanelDark", (0.12, 0.12, 0.13), 0.6), material("RadiatorWhite", (0.92, 0.93, 0.94), 0.25)
# the shutter is dark and the fins bright: in low polar sun a light shutter's slat seams read as fins (site_check_2)
m_cover, m_blue, m_yel = material("ShutterCharcoal", (0.07, 0.075, 0.085), 0.6, 0.3), material("CoverButton", (0.05, 0.2, 0.8), 0.35), yellow()
RAD_W, RAD_Z0, RAD_Z1, RAD_X = 0.60, 0.45, 1.35, 0.10  # fin panel width (y), bottom, top, front face x
box("radiator_frame", (0.04, RAD_W + 0.04, RAD_Z1 - RAD_Z0 + 0.04), (RAD_X - 0.02, 0.0, (RAD_Z0 + RAD_Z1) / 2), m_alu)
for j in range(11):
    box(f"fin_{j}", (0.05, 0.008, RAD_Z1 - RAD_Z0), (RAD_X + 0.025, -RAD_W / 2 + (j + 0.5) * RAD_W / 11, (RAD_Z0 + RAD_Z1) / 2), m_fin, bevel=0)
box("standoff", (RAD_X - 0.04, 0.2, 0.3), ((RAD_X - 0.04) / 2, 0.0, (RAD_Z0 + RAD_Z1) / 2), m_alu)
# the dust cover is a roll-up shutter: its housing across the top is always there; `cover_closed` is the shutter drawn
# down over the fins (visual only, no collider -- processing.yaml `no_collision`), hidden by the task when it is open
COVER_H = RAD_Z1 - RAD_Z0 + 0.04
box("shutter_housing", (0.12, RAD_W + 0.1, 0.12), (RAD_X + 0.03, 0.0, RAD_Z1 + 0.08), m_alu)
box("cover_closed", (0.01, RAD_W + 0.04, COVER_H), (RAD_X + 0.06, 0.0, RAD_Z1 + 0.02 - COVER_H / 2), m_cover, bevel=0.002)
for sy in (-1, 1):
    box(f"guide_rail_{sy}", (0.03, 0.02, RAD_Z1 - RAD_Z0 + 0.08), (RAD_X + 0.05, sy * (RAD_W / 2 + 0.03), (RAD_Z0 + RAD_Z1) / 2), m_alu, bevel=0)
# cover actuator box beside the radiator: its face carries the button like the service panel's, same mushroom
ACT_Y, BTN_Z = RAD_W / 2 + 0.22, 0.78
box("actuator_box", (0.02, 0.24, 0.30), (0.01, ACT_Y, BTN_Z + 0.05), m_yel)
box("actuator_post", (0.06, 0.06, BTN_Z - 0.1), (0.01, ACT_Y, (BTN_Z - 0.1) / 2), m_alu)
PLATE_FACE, BTN_COLLAR_R, BTN_COLLAR_X, BTN_CAP_R, BTN_CAP_X = 0.02, 0.036, 0.03, 0.032, 0.06
cylinder("button_collar", BTN_COLLAR_R, 0.02, (BTN_COLLAR_X, ACT_Y, BTN_Z), "X", m_dark)
cylinder("button_stem", 0.018, 0.02, (0.05, ACT_Y, BTN_Z), "X", m_dark)
dome("button_cap", BTN_CAP_R, (BTN_CAP_X, ACT_Y, BTN_Z), "X", m_blue)
export("pdu_radiator", dict(
    cover_button=dict(type="press", xyz=[BTN_CAP_X + BTN_CAP_R, ACT_Y, BTN_Z], normal=[1, 0, 0],
                      interface=dict(kind="mushroom_button", cap_radius_m=BTN_CAP_R, cap_centre_height_m=BTN_CAP_X - PLATE_FACE,
                                     collar_radius_m=BTN_COLLAR_R, collar_top_height_m=BTN_COLLAR_X + 0.01 - PLATE_FACE, travel_m=0.01)),
    radiator=dict(type="indicator", xyz=[RAD_X + 0.07, 0.0, (RAD_Z0 + RAD_Z1) / 2], normal=[1, 0, 0]),
))

# ---- per-equipment interaction parts (research repo docs/heldout_design.md §9): each unit type has its own look ------
PLATE_FACE, BTN_COLLAR_R, BTN_COLLAR_X, BTN_CAP_R, BTN_CAP_X = 0.02, 0.036, 0.03, 0.032, 0.06
BUTTON_INTERFACE = dict(kind="mushroom_button", cap_radius_m=BTN_CAP_R, cap_centre_height_m=BTN_CAP_X - PLATE_FACE,
                        collar_radius_m=BTN_COLLAR_R, collar_top_height_m=BTN_COLLAR_X + 0.01 - PLATE_FACE, travel_m=0.01)


def mushroom(y, z, m_cap, m_dark):
    cylinder("button_collar", BTN_COLLAR_R, 0.02, (BTN_COLLAR_X, y, z), "X", m_dark)
    cylinder("button_stem", 0.018, 0.02, (0.05, y, z), "X", m_dark)
    dome("button_cap", BTN_CAP_R, (BTN_CAP_X, y, z), "X", m_cap)


def control_plate(name, plate, border, cap, decor):
    """A unit's control face, mounted on its enclosure's front (x = 0): status lamp over a mushroom reset button, the
    same interaction heights as the PDU panel (button 0.78 m, lamp 1.02 m) so every unit is pressed alike. The reset
    cap is never red or green -- those are the lamp's two states, and a red cap under a green lamp was read as the lamp
    (site_check_7); IEC 60204-1 §10.2.1 gives reset white, grey, black or blue."""
    reset()
    m_plate, m_border, m_cap = material(*plate), material(*border), material(*cap)  # after reset(), which clears materials
    m_dark = material("PanelDark", (0.12, 0.12, 0.13), 0.6)
    m_lamp = material("StatusLamp", (0.2, 0.9, 0.25), 0.2, emission=(0.1, 1.0, 0.2))
    box("plate", (0.02, 0.30, 0.46), (0.01, 0.0, 0.86), m_plate)
    for sy in (-1, 1):
        box(f"border_{sy}", (0.022, 0.012, 0.46), (0.011, sy * 0.144, 0.86), m_border, bevel=0)
    mushroom(0.0, 0.78, m_cap, m_dark)
    cylinder("lamp_base", 0.04, 0.02, (0.03, 0.0, 1.02), "X", m_dark)
    dome("status_lamp", 0.034, (0.04, 0.0, 1.02), "X", m_lamp)
    decor(m_dark)
    export(name, dict(press=dict(type="press", xyz=[BTN_CAP_X + BTN_CAP_R, 0.0, 0.78], normal=[1, 0, 0], interface=BUTTON_INTERFACE),
                      lamp=dict(type="indicator", xyz=[0.074, 0.0, 1.02], normal=[1, 0, 0], material="StatusLamp")))


def led_bar(m_dark):  # battery management: a state-of-charge LED bar under the lamp (white: green is the lamp's)
    m_led = material("SocLed", (0.85, 0.9, 0.95), 0.3, emission=(0.5, 0.55, 0.6))
    for j in range(5):
        box(f"soc_led_{j}", (0.008, 0.028, 0.014), (0.024, -0.07 + j * 0.035, 0.92), m_led if j < 4 else m_dark, bevel=0)


def gauge(m_dark):  # ISRU plant: an outlet-pressure readout under the lamp -- a dark rectangular display, not a round
    # white dial: from the inspection view a white dial beside the lamp was read as the lamp (site_check_8)
    box("display_bezel", (0.02, 0.16, 0.07), (0.03, 0.0, 0.92), m_dark, bevel=0.003)
    box("display_face", (0.004, 0.13, 0.045), (0.041, 0.0, 0.92), material("DisplayFace", (0.05, 0.12, 0.14), 0.2, emission=(0.02, 0.1, 0.12)), bevel=0)


control_plate("bms_panel", ("Graphite", (0.16, 0.17, 0.19), 0.5, 0.4), ("BmsTeal", (0.1, 0.45, 0.5), 0.5),
              ("ResetWhite", (0.9, 0.9, 0.88), 0.35), led_bar)
control_plate("plant_panel", ("PlantWhite", (0.88, 0.88, 0.86), 0.5), ("PlantOrange", (0.95, 0.42, 0.05), 0.45),
              ("ResetBlack", (0.06, 0.06, 0.07), 0.35), gauge)


def cradle(name, frame, decor):
    """A unit's module cradle: the shelf's tray (so a module is picked and placed there as a spare is on a shelf), dressed
    for its unit."""
    reset()
    tray(material(*frame))
    decor()
    export(name, dict(tray=dict(TRAY_POINT, role="equipment_bay"), top_z=SHELF_Z + 0.03))


def hv_dress():  # battery: orange/black high-voltage border, a busbar block with two terminals behind the module
    m_or, m_blk, m_cu = material("HvOrange", (0.95, 0.4, 0.02), 0.45), material("HvBlack", (0.05, 0.05, 0.05), 0.6), material("Copper", (0.72, 0.42, 0.2), 0.3, 0.9)
    for j in range(9):
        box(f"border_{j}", (0.012, SHELF_W / 9, 0.02), (SHELF_D + 0.006, -SHELF_W / 2 + (j + 0.5) * SHELF_W / 9, SHELF_Z - 0.01), m_or if j % 2 == 0 else m_blk, bevel=0)
    box("busbar_block", (0.025, 0.40, 0.14), (0.02, 0.0, SHELF_Z + 0.08), m_blk)
    for sy in (-1, 1):
        cylinder(f"terminal_{sy}", 0.018, 0.03, (0.047, sy * 0.1, SHELF_Z + 0.1), "X", m_cu)


def filter_dress():  # ISRU filter: a white housing plate with a blue inlet/outlet manifold behind the cartridge
    m_blue, m_wh = material("ProcessBlue", (0.08, 0.25, 0.6), 0.4, 0.3), material("PlantWhite", (0.88, 0.88, 0.86), 0.5)
    box("manifold_plate", (0.02, 0.40, 0.26), (0.02, 0.0, SHELF_Z + 0.13), m_wh)
    for sy in (-1, 1):
        cylinder(f"port_{sy}", 0.03, 0.04, (0.05, sy * 0.1, SHELF_Z + 0.12), "X", m_blue)
        cylinder(f"pipe_{sy}", 0.03, 0.5, (0.02, sy * 0.1, SHELF_Z - 0.2), "Z", m_blue)


cradle("battery_cradle", ("CradleDark", (0.2, 0.21, 0.23), 0.5, 0.6), hv_dress)
cradle("filter_cradle", ("CradleSteel", (0.55, 0.57, 0.6), 0.45, 0.7), filter_dress)

# ---- HVAC condenser of the battery storage unit, with its dust shutter and cover button --------------------------------
reset()
m_body, m_dark, m_grill = material("CondenserBeige", (0.78, 0.76, 0.7), 0.55), material("PanelDark", (0.12, 0.12, 0.13), 0.6), material("GrilleGrey", (0.3, 0.31, 0.33), 0.5, 0.6)
m_cover, m_blue, m_yel = material("ShutterCharcoal", (0.07, 0.075, 0.085), 0.6, 0.3), material("CoverButton", (0.05, 0.2, 0.8), 0.35), yellow()
CD, CW, CZ0, CZ1 = 0.70, 1.10, 0.12, 1.30  # depth (x), width (y), body bottom/top; front face at x = +CD/2
box("body", (CD, CW, CZ1 - CZ0), (0.0, 0.0, (CZ0 + CZ1) / 2), m_body)
for sx in (-1, 1):
    for sy in (-1, 1):
        box(f"foot_{sx}{sy}", (0.08, 0.08, CZ0), (sx * (CD / 2 - 0.06), sy * (CW / 2 - 0.06), CZ0 / 2), m_dark)
for j, y in enumerate((-0.25, 0.25)):  # two fan openings on the front face, guards and blade hubs
    cylinder(f"fan_ring_{j}", 0.21, 0.02, (CD / 2 + 0.005, y, 0.78), "X", m_grill)
    cylinder(f"fan_hub_{j}", 0.05, 0.03, (CD / 2 + 0.01, y, 0.78), "X", m_dark)
    for k in range(4):
        box(f"fan_guard_{j}_{k}", (0.006, 0.40, 0.012), (CD / 2 + 0.018, y, 0.63 + k * 0.1), m_dark, bevel=0)
# the dust shutter over the fan face: visual only (processing.yaml `no_collision`), shown or hidden by the task
box("shutter_housing", (0.10, CW, 0.10), (CD / 2 + 0.02, 0.0, CZ1 - 0.05), m_dark)
box("cover_closed", (0.01, CW - 0.04, 0.62), (CD / 2 + 0.045, 0.0, CZ1 - 0.1 - 0.31), m_cover, bevel=0.002)
# the cover actuator: a yellow box on a post beside the unit, its blue mushroom at the common button height
ACT_Y = CW / 2 + 0.24
box("actuator_box", (0.02, 0.24, 0.30), (CD / 2 + 0.01, ACT_Y, 0.83), m_yel)
box("actuator_post", (0.06, 0.06, 0.68), (CD / 2 - 0.02, ACT_Y, 0.34), alu())
for part, geom in (("button_collar", (BTN_COLLAR_R, 0.02, (CD / 2 + BTN_COLLAR_X, ACT_Y, 0.78))), ("button_stem", (0.018, 0.02, (CD / 2 + 0.05, ACT_Y, 0.78)))):
    cylinder(part, *geom, "X", m_dark)
dome("button_cap", BTN_CAP_R, (CD / 2 + BTN_CAP_X, ACT_Y, 0.78), "X", m_blue)
export("hvac_condenser", dict(
    cover_button=dict(type="press", xyz=[CD / 2 + BTN_CAP_X + BTN_CAP_R, ACT_Y, 0.78], normal=[1, 0, 0], interface=BUTTON_INTERFACE),
    condenser=dict(type="indicator", xyz=[CD / 2 + 0.02, 0.0, 0.78], normal=[1, 0, 0]),
))

# ---- ISRU filter cartridge (dynamic item; processing adds the lifting handle and the serial label) ------------------------
reset()
m_can, m_cap, m_band = material("CanisterGrey", (0.32, 0.33, 0.35), 0.45, 0.6), alu(), material("FilterYellow", (0.95, 0.75, 0.1), 0.5)
R, H = 0.13, 0.34
cylinder("canister", R, H - 0.06, (0.0, 0.0, H / 2), "Z", m_can)
cylinder("end_cap_bottom", R + 0.008, 0.03, (0.0, 0.0, 0.015), "Z", m_cap)
cylinder("end_cap_top", R + 0.008, 0.03, (0.0, 0.0, H - 0.015), "Z", m_cap)
for k in range(3):
    cylinder(f"rib_{k}", R + 0.004, 0.012, (0.0, 0.0, 0.1 + k * 0.07), "Z", m_cap)
cylinder("label_band", R + 0.002, 0.05, (0.0, 0.0, H - 0.07), "Z", m_band)
export("filter_cartridge", dict(size=[2 * R, 2 * R, H]))
