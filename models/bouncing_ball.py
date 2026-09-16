import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]

    pos = state[0]
    vel = state[1]

    accel = -gravity

    state_derivative = np.array([vel, accel])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "mass": 1,  # point mass at end of rod (kg)
        "restitution_coeff": 0.7 # coefficient of restitution
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]

    pos = state[0]
    vel = state[1]

    kinetic_energy = 0.5 * mass * (vel ** 2)
    potential_energy = pos * gravity
    return kinetic_energy, potential_energy


# Detect a collision
def detect_event(t, state, params):
    return state[0] < 0


# Return the updated state after the collision
def handle_event(t, state, params):
    restitution_coeff = params["restitution_coeff"]
    pos = state[0]
    vel = state[1]

    return np.array([0, -vel * restitution_coeff])