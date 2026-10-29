"""Live representative Gazebo contact force for each simulated finger."""

import math
import time
import tkinter as tk
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ContactsState

rclpy.init()
node = Node("piper_sim_contact_display")
root = tk.Tk()
root.title("Gazebo - simulated contact force")
root.geometry("900x340+1300+120")
root.attributes("-topmost", True)
tk.Label(root, text="SIMULATED CONTACT FORCE", font=("Sans", 12, "bold")).pack(pady=4)
tk.Label(
    root, text="Largest reported contact-state force; not total grip force", font=("Sans", 8)
).pack()
values = {}
var = {}
for finger in ("gripper_link1", "gripper_link2"):
    var[finger] = tk.StringVar(value=f"{finger}: waiting for sensor")
    tk.Label(root, textvariable=var[finger], font=("Sans", 12), anchor="w").pack(
        fill="x", padx=20, pady=4
    )

    def callback(msg, name=finger):
        hits = [
            s
            for s in msg.states
            if "cylinder::" in s.collision1_name or "cylinder::" in s.collision2_name
        ]
        if not hits:
            values[name] = (time.monotonic(), 0.0, 0)
            return
        forces = [
            math.sqrt(
                s.total_wrench.force.x**2 + s.total_wrench.force.y**2 + s.total_wrench.force.z**2
            )
            for s in hits
        ]
        values[name] = (time.monotonic(), max(forces), len(hits))

    node.create_subscription(ContactsState, "/" + finger + "/contacts", callback, 10)
tk.Label(root, text="N = newtons   |   Gray = no recent sensor data", font=("Sans", 8)).pack(pady=4)


def tick():
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0)
    for finger in var:
        t, value, count = values.get(finger, (0, 0, 0))
        status = "NO DATA" if time.monotonic() - t > 1 else ("CONTACT" if count else "no contact")
        var[finger].set(f"{finger}:  {value:6.2f} N   [{status}]")
    root.after(80, tick)


def close():
    node.destroy_node()
    rclpy.shutdown()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", close)
root.after(0, tick)
root.mainloop()
