import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from integrators import rk4 as integrator

# Basic simulation of the pendulum

params = model.generate_params()


# some set-up
initial_state = np.array([np.pi / 8, 0.0])

timestep = 1e-3
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# simulation loop
for step, t in enumerate(time_traj[:-1]):
    state_traj[:, step + 1] = integrator.take_step(
        t=t,
        state=state_traj[:, step],
        params=params,
        model=model,
        timestep=timestep
        )
    if model.detect_event(t, state_traj[:, step + 1], params):
        state_traj[:, step + 1] = model.calculate_state_after_event(t, state_traj[:, step + 1], params)

# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

plt.figure()
# plt.plot(time_traj, potential_energy, label="Potential energy")
# plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
# plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
# plt.xlabel("Time (s)")
# plt.ylabel("Energy (J)")
# plt.title("Pendulum energy")

plt.plot(time_traj, state_traj[0], label="angle")
plt.plot(time_traj, state_traj[1], label="ang vel")
plt.xlabel("Time (s)")
plt.ylabel("rad or rad/s")
plt.title("state trajectory")

plt.legend()
plt.tight_layout()
plt.show()

# TODO: make a phase portrait plot
