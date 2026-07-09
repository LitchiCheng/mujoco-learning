import mujoco
from mujoco.usd import exporter

m = mujoco.MjModel.from_xml_path('./model/franka_emika_panda/scene.xml')
d = mujoco.MjData(m)

# Create the USDExporter
exp = exporter.USDExporter(model=m)

duration = 5
framerate = 60
while d.time < duration:

  # Step the physics
  mujoco.mj_step(m, d)

  if exp.frame_count < d.time * framerate:
    # Update the USD with a new frame
    exp.update_scene(data=d)

# Export the USD file
exp.save_scene(filetype="usd")