"""Generate the contact-grasp scene from the official Piper model.

Objects are parameterized by shape and dimensions through PIPER_SIM_*
variables; the Gazebo model name stays "cylinder" because contact, state and
evaluation code depend on it.
"""

from pathlib import Path
import json
import math
import os
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

root_path = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
min_contact_depth = float(os.environ.get("PIPER_SIM_MIN_CONTACT_DEPTH", "0.001"))
# Position PID used by each simulated prismatic finger.
finger_kp = float(os.environ.get("PIPER_SIM_FINGER_KP", "2000"))
finger_kd = float(os.environ.get("PIPER_SIM_FINGER_KD", "10"))
finger_ki = 0
friction = float(os.environ.get("PIPER_SIM_FRICTION", "2"))
step_size = float(os.environ.get("PIPER_SIM_STEP_SIZE", "0.001"))
solver_iters = (
    int(os.environ["PIPER_SIM_SOLVER_ITERS"]) if "PIPER_SIM_SOLVER_ITERS" in os.environ else None
)
finger_geometry = os.environ.get("PIPER_SIM_FINGER_GEOMETRY", "mesh")
object_shape = os.environ.get("PIPER_SIM_OBJECT_SHAPE", "cylinder")
object_mass = float(os.environ.get("PIPER_SIM_OBJECT_MASS", "0.06"))
object_dims = os.environ.get("PIPER_SIM_OBJECT_DIMS", "")
object_yaw = float(os.environ.get("PIPER_SIM_OBJECT_YAW_DEG", "0"))
object_friction = float(os.environ.get("PIPER_SIM_OBJECT_FRICTION", str(friction)))
object_mesh = os.environ.get("PIPER_SIM_OBJECT_MESH", "")
rgbd = os.environ.get("PIPER_SIM_RGBD", "0") == "1"
occluder = os.environ.get("PIPER_SIM_OCCLUDER", "0") == "1"
view_index = int(os.environ.get("PIPER_SIM_VIEW", "0"))
object_x = float(os.environ.get("PIPER_SIM_OBJECT_X", "0.45"))
object_y = float(os.environ.get("PIPER_SIM_OBJECT_Y", "0"))
if not math.isfinite(object_x) or not math.isfinite(object_y) or not 0.3 <= object_x <= 0.55 or abs(object_y) > 0.12:
    raise ValueError("Simulated object position is outside the supported test area")
control_rate = int(os.environ.get("PIPER_SIM_CONTROL_RATE", "200"))
arm_pid = os.environ.get("PIPER_SIM_ARM_PID", "0") == "1"
arm_kp = float(os.environ.get("PIPER_SIM_ARM_KP", "50"))
arm_kd = float(os.environ.get("PIPER_SIM_ARM_KD", "2"))
if not 10 <= control_rate <= 1000:
    raise ValueError("Invalid control update rate")
SHAPES = ("cylinder", "cube", "sphere", "box", "bar", "mesh")
if object_shape not in SHAPES or not 0.001 <= object_mass <= 1:
    raise ValueError(f"Invalid object shape (choose from {SHAPES}) or mass")
if view_index not in (0, 1, 2):
    raise ValueError("PIPER_SIM_VIEW must be 0, 1 or 2")
if not math.isfinite(object_yaw) or abs(object_yaw) > 180:
    raise ValueError("PIPER_SIM_OBJECT_YAW_DEG must be finite within +/-180")
if not 0.1 <= object_friction <= 10:
    raise ValueError("PIPER_SIM_OBJECT_FRICTION must be between 0.1 and 10")


def parse_dims(text, count, default):
    """Parse a comma-separated dimension list with an exact element count."""
    values = [float(part) for part in text.split(",")] if text.strip() else list(default)
    if len(values) != count or not all(math.isfinite(v) and 0.005 <= v <= 0.3 for v in values):
        raise ValueError(f"Object dimensions {text!r} must be {count} values in [0.005, 0.3] m")
    return values


if object_shape == "cylinder":
    diameter, height = parse_dims(object_dims, 2, [0.046, 0.08])
    dims = [diameter, height]
elif object_shape == "sphere":
    (diameter,) = parse_dims(object_dims, 1, [0.04])
    dims = [diameter]
elif object_shape in ("cube", "box", "bar"):
    default = {"cube": [0.046, 0.046, 0.046],
               "box": [0.05, 0.04, 0.03],
               "bar": [0.09, 0.02, 0.02]}[object_shape]
    dx, dy, dz = parse_dims(object_dims, 3, default)
    dims = [dx, dy, dz]
elif object_shape == "mesh":
    if not object_mesh:
        raise ValueError("PIPER_SIM_OBJECT_MESH is required for the mesh shape")
    mesh_path = Path(object_mesh)
    if not mesh_path.is_file():
        raise ValueError(f"Object mesh file does not exist: {mesh_path}")


def mesh_aabb(path):
    """Axis-aligned bounding box of a binary/ASCII STL or OBJ mesh, in metres."""
    data = path.read_bytes()
    vertices = []
    if path.suffix.lower() == ".obj":
        for line in data.decode("ascii", errors="replace").splitlines():
            parts = line.split()
            if parts[:1] == ["v"] and len(parts) >= 4:
                vertices.append([float(v) for v in parts[1:4]])
    elif data[:5] == b"solid" and b"facet" in data[:2048]:
        for line in data.decode("ascii", errors="replace").splitlines():
            parts = line.split()
            if parts[:1] == ["vertex"] and len(parts) >= 4:
                vertices.append([float(v) for v in parts[1:4]])
    elif len(data) >= 84:
        (count,) = struct.unpack_from("<I", data, 80)
        if len(data) >= 84 + 50 * count:
            for i in range(count):
                values = struct.unpack_from("<12f", data, 84 + 50 * i)
                vertices += [values[3:6], values[6:9], values[9:12]]
    if not vertices:
        raise ValueError(f"Unsupported or empty object mesh: {path}")
    low = [min(v[i] for v in vertices) for i in range(3)]
    high = [max(v[i] for v in vertices) for i in range(3)]
    return [high[i] - low[i] for i in range(3)], len(vertices)


if object_shape == "mesh":
    mesh_size, mesh_vertex_count = mesh_aabb(mesh_path)
    dims = mesh_size
    if any(not 0.005 <= v <= 0.3 for v in dims):
        raise ValueError(f"Mesh extents {dims} are outside the supported [0.005, 0.3] m range")

if object_shape == "cylinder":
    height = dims[1]
    center_z = -0.15 + height / 2 + 0.001
elif object_shape == "sphere":
    height = dims[0]
    center_z = -0.15 + height / 2 + 0.001
else:
    height = dims[2]
    center_z = -0.15 + height / 2 + 0.001
if not 0 <= min_contact_depth <= 0.001:
    raise ValueError("PIPER_SIM_MIN_CONTACT_DEPTH must be between 0 and 0.001 m")
if not 0 <= finger_kd <= 100:
    raise ValueError("PIPER_SIM_FINGER_KD must be between 0 and 100")
if not 1 <= finger_kp <= 2000:
    raise ValueError("PIPER_SIM_FINGER_KP must be between 1 and 2000")
if not 0.1 <= friction <= 10:
    raise ValueError("PIPER_SIM_FRICTION must be between 0.1 and 10")
if not 0.0001 <= step_size <= 0.001 or (solver_iters is not None and not 1 <= solver_iters <= 500):
    raise ValueError("Invalid simulation step or solver iteration count")
if finger_geometry not in ("mesh", "box"):
    raise ValueError("PIPER_SIM_FINGER_GEOMETRY must be mesh or box")
subprocess.run(
    [sys.executable, str(root_path / "sim/gazebo/make_robot.py"), str(out / "robot.urdf")] + (["--rgbd"] if rgbd else []),
    check=True,
)
r = ET.parse(out / "robot.urdf").getroot()
if finger_geometry == "box":
    # A simplified flat contact pad for an isolated solver experiment only.
    # It is not the real finger geometry and must not be used for real planning.
    for name in ("gripper_link1", "gripper_link2"):
        collision = r.find(f"link[@name='{name}']/collision")
        origin = collision.find("origin")
        if origin is None:
            origin = ET.SubElement(collision, "origin")
        origin.set("xyz", "0 -0.035 0.01325")
        origin.set("rpy", "0 0 0")
        geometry = collision.find("geometry")
        geometry.clear()
        ET.SubElement(geometry, "box", {"size": "0.0254 0.05 0.0265"})


def add(parent, tag, attributes=None, text=None):
    """Append an XML element and return it for building the scene tree."""
    element = ET.SubElement(parent, tag, attributes or {})
    element.text = text
    return element


for g in r.findall("gazebo"):
    for p in list(g.findall("plugin")):
        g.remove(p)
    # Leave arm gravity disabled as in the existing model. Object gravity stays on.
    if not rgbd:
        for sensor in g.findall("sensor"):
            g.remove(sensor)
dummy = r.find("link[@name='gripper_link']")
inertial = add(dummy, "inertial")
add(inertial, "mass", {"value": "0.01"})
add(
    inertial,
    "inertia",
    {"ixx": "1e-8", "iyy": "1e-8", "izz": "1e-8", "ixy": "0", "ixz": "0", "iyz": "0"},
)
arm_system = add(r, "ros2_control", {"name": "GazeboArm", "type": "system"})
add(add(arm_system, "hardware"), "plugin", text="gazebo_ros2_control/GazeboSystem")
hand_system = add(r, "ros2_control", {"name": "GazeboHand", "type": "system"})
add(add(hand_system, "hardware"), "plugin", text="gazebo_ros2_control/GazeboSystem")
initial = ([0.0, 2.8, -2.4, 0.0, -0.1, 0.0] if rgbd else
           [0.0, 0.8, -0.8, 0.0, 0.8, 0.0])
# Camera-view variants: base yaw shifts the wrist-camera azimuth so the same
# object is seen from a different angle; only used with the RGBD pose.
if rgbd and view_index == 1:
    initial[0] = 0.3
elif rgbd and view_index == 2:
    initial[0] = -0.3
for name, value in list(zip([f"joint{i}" for i in range(1, 7)], initial)) + [
    ("gripper", 0.07),
    ("gripper_joint1", 0.035),
    ("gripper_joint2", -0.035),
]:
    j = add(arm_system if name.startswith("joint") else hand_system, "joint", {"name": name})
    if arm_pid and name.startswith("joint"):
        # Wrist links have much less inertia than the shoulder and elbow.
        wrist = name in ("joint4", "joint5", "joint6")
        add(j, "param", {"name": "pos_kp"}, str(3.0 if wrist else arm_kp))
        add(j, "param", {"name": "pos_kd"}, str(0.04 if wrist else arm_kd))
        add(j, "param", {"name": "pos_ki"}, "0")
    if name == "gripper":
        add(j, "param", {"name": "pos_kp"}, "100")
        add(j, "param", {"name": "pos_kd"}, "2")
        add(j, "param", {"name": "pos_ki"}, "0")
    if name.startswith("gripper_joint"):
        # Drive each finger separately in Gazebo; keep real robot files unchanged.
        original = r.find(f"joint[@name='{name}']")
        original.remove(original.find("mimic"))
        add(j, "param", {"name": "pos_kp"}, str(finger_kp))
        add(j, "param", {"name": "pos_kd"}, str(finger_kd))
        add(j, "param", {"name": "pos_ki"}, str(finger_ki))
    add(
        j,
        "command_interface",
        {
            "name": (
                "position_pid"
                if name.startswith("gripper_joint") or (arm_pid and name.startswith("joint"))
                else "position"
            )
        },
    )
    add(
        add(j, "state_interface", {"name": "position"}),
        "param",
        {"name": "initial_value"},
        str(value),
    )
    add(j, "state_interface", {"name": "velocity"})
g = add(r, "gazebo")
p = add(g, "plugin", {"name": "gazebo_ros2_control", "filename": "libgazebo_ros2_control.so"})
controller_config = (
    (root_path / "sim/gazebo/grasp/controllers.yaml")
    .read_text()
    .replace("    update_rate: 200", f"    update_rate: {control_rate}", 1)
)
(out / "controllers.yaml").write_text(controller_config)
add(p, "parameters", text=str(out / "controllers.yaml"))
for link in ["gripper_link1", "gripper_link2"]:
    g = add(r, "gazebo", {"reference": link})
    add(g, "mu1", text=str(friction))
    add(g, "mu2", text=str(friction))
    add(g, "kp", text="100000")
    add(g, "kd", text="10")
    add(g, "minDepth", text=str(min_contact_depth))
    add(g, "maxVel", text="0.1")
    sensor = add(g, "sensor", {"name": link + "_contact", "type": "contact"})
    add(sensor, "always_on", text="true")
    add(sensor, "update_rate", text="100")
    add(add(sensor, "contact"), "collision", text=link + "_collision")
    bumper = add(
        sensor, "plugin", {"name": link + "_bumper", "filename": "libgazebo_ros_bumper.so"}
    )
    ros = add(bumper, "ros")
    add(ros, "remapping", text="bumper_states:=/" + link + "/contacts")
    add(bumper, "frame_name", text=link)
ET.ElementTree(r).write(out / "robot.urdf")
w = ET.Element("sdf", {"version": "1.6"})
world = add(w, "world", {"name": "default"})
physics = add(world, "physics", {"name": "ode", "type": "ode"})
add(physics, "max_step_size", text=str(step_size))
add(physics, "real_time_update_rate", text="1000")
if solver_iters is not None:
    ode = add(physics, "ode")
    solver = add(ode, "solver")
    add(solver, "iters", text=str(solver_iters))
add(world, "gravity", text="0 0 -9.81")
light = add(world, "light", {"name": "sun", "type": "directional"})
add(light, "pose", text="0 0 10 0 0 0")
add(light, "diffuse", text=".8 .8 .8 1")
add(light, "direction", text="-.2 .3 -1")
gui = add(world, "gui")
cam = add(gui, "camera", {"name": "user_camera"})
add(cam, "pose", text="1 -1 .7 0 .35 2.4")
ground = add(world, "model", {"name": "ground"})
add(ground, "static", text="true")
add(ground, "pose", text="0 0 -.15 0 0 0")
l = add(ground, "link", {"name": "ground"})
for typ in ["collision", "visual"]:
    e = add(l, typ, {"name": typ})
    plane = add(add(e, "geometry"), "plane")
    add(plane, "normal", text="0 0 1")
    add(plane, "size", text="10 10")


def object_inertia(shape, mass, dims):
    """Diagonal inertia for each supported primitive; mesh uses its AABB box."""
    if shape == "cylinder":
        radius, height = dims[0] / 2, dims[1]
        ixx = mass * (3 * radius**2 + height**2) / 12
        return [ixx, ixx, mass * radius**2 / 2]
    if shape == "sphere":
        return [mass * dims[0] ** 2 / 10] * 3
    dx, dy, dz = dims
    return [mass * (dy**2 + dz**2) / 12,
            mass * (dx**2 + dz**2) / 12,
            mass * (dx**2 + dy**2) / 12]


# Keep the model name for the existing state and contact subscriptions.
obj = add(world, "model", {"name": "cylinder"})
add(obj, "pose", text=f"{object_x} {object_y} {center_z} 0 0 {math.radians(object_yaw)}")
l = add(obj, "link", {"name": "body"})
inertial = add(l, "inertial")
add(inertial, "mass", text=str(object_mass))
i = add(inertial, "inertia")
inertia_values = object_inertia(object_shape, object_mass, dims)
for k, v in zip(("ixx", "iyy", "izz"), inertia_values):
    add(i, k, text=str(v))
for k in ("ixy", "ixz", "iyz"):
    add(i, k, text="0")
e = add(l, "collision", {"name": "cylinder_collision"})
geometry = add(e, "geometry")
if object_shape == "cylinder":
    geo = add(geometry, "cylinder")
    add(geo, "radius", text=str(dims[0] / 2))
    add(geo, "length", text=str(dims[1]))
elif object_shape == "sphere":
    geo = add(geometry, "sphere")
    add(geo, "radius", text=str(dims[0] / 2))
elif object_shape == "mesh":
    geo = add(geometry, "mesh")
    add(geo, "uri", text=f"file://{mesh_path.resolve()}")
else:
    add(add(geometry, "box"), "size", text=" ".join(str(v) for v in dims))
fr = add(add(add(e, "surface"), "friction"), "ode")
add(fr, "mu", text=str(object_friction))
add(fr, "mu2", text=str(object_friction))


def red_visual(name, geometry_element, pose=None):
    e = add(l, "visual", {"name": name})
    if pose:
        add(e, "pose", text=pose)
    e.append(geometry_element)
    m = add(e, "material")
    add(m, "ambient", text=".9 .02 .02 1")
    add(m, "diffuse", text=".9 .02 .02 1")


if object_shape == "cylinder":
    # Legacy red side band with white caps, as in the earlier fixed scenes.
    for name, z, length, color in [
        ("red", 0, max(0.0, dims[1] - 0.02), ".9 .02 .02 1"),
        ("top", dims[1] / 2 - 0.005, 0.01, ".95 .95 .95 1"),
        ("bottom", -dims[1] / 2 + 0.005, 0.01, ".95 .95 .95 1"),
    ]:
        e = add(l, "visual", {"name": name})
        add(e, "pose", text=f"0 0 {z} 0 0 0")
        geo = add(add(e, "geometry"), "cylinder")
        add(geo, "radius", text=str(dims[0] / 2))
        add(geo, "length", text=str(length))
        m = add(e, "material")
        add(m, "ambient", text=color)
        add(m, "diffuse", text=color)
elif object_shape == "sphere":
    geo = add(ET.Element("tmp"), "geometry")
    sphere = add(geo, "sphere")
    add(sphere, "radius", text=str(dims[0] / 2))
    red_visual("red", geo)
elif object_shape == "mesh":
    geo = add(ET.Element("tmp"), "geometry")
    mesh = add(geo, "mesh")
    add(mesh, "uri", text=f"file://{mesh_path.resolve()}")
    red_visual("red", geo)
else:
    geo = add(ET.Element("tmp"), "geometry")
    add(add(geo, "box"), "size", text=" ".join(str(v) for v in dims))
    red_visual("red", geo)
if occluder:
    # A white blocker beside the object partially occludes it from tilted
    # wrist views. It is static and deliberately not the target colour.
    occ = add(world, "model", {"name": "occluder"})
    add(occ, "static", text="true")
    add(occ, "pose", text=f"{object_x + 0.055} {object_y - 0.04} {-0.15 + 0.03} 0 0 0")
    ol = add(occ, "link", {"name": "blocker"})
    oe = add(ol, "collision", {"name": "blocker_collision"})
    add(add(add(oe, "geometry"), "box"), "size", text="0.05 0.02 0.06")
    add(add(add(add(oe, "surface"), "friction"), "ode"), "mu", text="1.0")
    ove = add(ol, "visual", {"name": "blocker_visual"})
    add(add(add(ove, "geometry"), "box"), "size", text="0.05 0.02 0.06")
    om = add(ove, "material")
    add(om, "ambient", text=".93 .93 .93 1")
    add(om, "diffuse", text=".93 .93 .93 1")
p = add(world, "plugin", {"name": "state", "filename": "libgazebo_ros_state.so"})
add(add(p, "ros"), "namespace", text="/gazebo")
add(p, "update_rate", text="30")
ET.ElementTree(w).write(out / "grasp.world")
(out / "scene_parameters.json").write_text(
    json.dumps(
        {
            "min_contact_depth_m": min_contact_depth,
            "finger_kp": finger_kp,
            "finger_kd": finger_kd,
            "friction": friction,
            "object_friction": object_friction,
            "step_size_s": step_size,
            "solver_iterations": solver_iters,
            "control_update_rate_hz": control_rate,
            "arm_position_pid": arm_pid,
            "arm_kp": arm_kp,
            "arm_kd": arm_kd,
            "finger_geometry": finger_geometry,
            "object_shape": object_shape,
            "object_dims_m": dims,
            "object_yaw_deg": object_yaw,
            "object_mesh": str(mesh_path) if object_shape == "mesh" else None,
            "object_mass_kg": object_mass,
            "occluder": occluder,
            "view_index": view_index,
            "rgbd_enabled": rgbd,
            "object_height_m": height,
            "object_center_z_m": center_z,
        },
        indent=2,
    )
    + "\n"
)
