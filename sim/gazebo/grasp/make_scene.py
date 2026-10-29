"""Generate the contact-grasp scene from the official Piper model."""

from pathlib import Path
import json
import os
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
control_rate = int(os.environ.get("PIPER_SIM_CONTROL_RATE", "200"))
arm_pid = os.environ.get("PIPER_SIM_ARM_PID", "0") == "1"
arm_kp = float(os.environ.get("PIPER_SIM_ARM_KP", "50"))
arm_kd = float(os.environ.get("PIPER_SIM_ARM_KD", "2"))
if not 10 <= control_rate <= 1000:
    raise ValueError("Invalid control update rate")
if object_shape not in ("cylinder", "cube") or not 0.001 <= object_mass <= 1:
    raise ValueError("Invalid object shape or mass")
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
    [sys.executable, str(root_path / "sim/gazebo/make_robot.py"), str(out / "robot.urdf")],
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
initial = [0.0, 0.8, -0.8, 0.0, 0.8, 0.0]
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
# Keep the model name for the existing state and contact subscriptions.
height = 0.046 if object_shape == "cube" else 0.08
center_z = -0.15 + height / 2 + 0.001
obj = add(world, "model", {"name": "cylinder"})
add(obj, "pose", text=f".45 0 {center_z} 0 0 0")
l = add(obj, "link", {"name": "body"})
inertial = add(l, "inertial")
add(inertial, "mass", text=str(object_mass))
i = add(inertial, "inertia")
if object_shape == "cube":
    inertia = object_mass * 0.046**2 / 6
    values = [("ixx", inertia), ("iyy", inertia), ("izz", inertia)]
else:
    values = [
        ("ixx", 0.00004 * object_mass / 0.06),
        ("iyy", 0.00004 * object_mass / 0.06),
        ("izz", 0.000016 * object_mass / 0.06),
    ]
for k, v in values + [("ixy", 0), ("ixz", 0), ("iyz", 0)]:
    add(i, k, text=str(v))
e = add(l, "collision", {"name": "cylinder_collision"})
geometry = add(e, "geometry")
if object_shape == "cube":
    add(add(geometry, "box"), "size", text=".046 .046 .046")
else:
    geo = add(geometry, "cylinder")
    add(geo, "radius", text=".023")
    add(geo, "length", text=".08")
fr = add(add(add(e, "surface"), "friction"), "ode")
add(fr, "mu", text=str(friction))
add(fr, "mu2", text=str(friction))
if object_shape == "cube":
    e = add(l, "visual", {"name": "red"})
    add(add(add(e, "geometry"), "box"), "size", text=".046 .046 .046")
    m = add(e, "material")
    add(m, "ambient", text=".9 .02 .02 1")
    add(m, "diffuse", text=".9 .02 .02 1")
else:
    for name, z, length, color in [
        ("red", 0, 0.06, ".9 .02 .02 1"),
        ("top", 0.035, 0.01, ".95 .95 .95 1"),
        ("bottom", -0.035, 0.01, ".95 .95 .95 1"),
    ]:
        e = add(l, "visual", {"name": name})
        add(e, "pose", text=f"0 0 {z} 0 0 0")
        geo = add(add(e, "geometry"), "cylinder")
        add(geo, "radius", text=".023")
        add(geo, "length", text=str(length))
        m = add(e, "material")
        add(m, "ambient", text=color)
        add(m, "diffuse", text=color)
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
            "step_size_s": step_size,
            "solver_iterations": solver_iters,
            "control_update_rate_hz": control_rate,
            "arm_position_pid": arm_pid,
            "arm_kp": arm_kp,
            "arm_kd": arm_kd,
            "finger_geometry": finger_geometry,
            "object_shape": object_shape,
            "object_mass_kg": object_mass,
            "object_height_m": height,
            "object_center_z_m": center_z,
        },
        indent=2,
    )
    + "\n"
)
