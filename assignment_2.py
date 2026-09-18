from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from integrators import rk4 as integrator
from models import inverted_pendulum_walker as model

# Fixed controls for this visualization example.
params = model.generate_params()


# Control

# Choose spoke angle: returns the modified params dict
def calculate_spoke_angle(t, state, params):
    return params

# Choose ankle torque returns the modified params dict
def calculate_ankle_torque(t, state, params):
    # Linearize about upright eq. point
    # ddtheta = (g/L)*theta + ((-2g/L)*theta - b*dtheta)) = (-g/L)*theta - b*dtheta
    m = params["mass"]
    g = params["gravity"]
    l = params["length"]
    b = 0.1     # damping parameter

    angle = state[0]
    ang_vel = state[1]
    target_torque = ((-2*g/l) * angle - b * ang_vel) * m * l
    params["ankle_torque"] = np.clip(target_torque, -0.1*m*g*l, 0.05*m*g*l)
    return params


initial_state = np.array([0.1, 0.0])
timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0

# Simulation loop. Replace this Euler step with your own integrator as needed.
for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]
    params = calculate_ankle_torque(t, state, params)
    next_state = integrator.take_step(t=t, state=state, params=params, model=model, timestep=timestep)

    if model.event_guard(state, next_state, params):
        next_state = model.event_dynamics(next_state, params)
        completed_steps += 1

    state_traj[:, step + 1] = next_state
    if completed_steps == desired_number_of_steps:
        break

time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()
