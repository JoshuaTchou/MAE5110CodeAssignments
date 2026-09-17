import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def generate_params(num_spokes = 8):
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "num_spokes": num_spokes,
        "alpha": (2 * np.pi) / (2 * num_spokes),
        "gamma": (1/8) * np.pi  # slope angle
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy


def detect_event(t, state, params):
    angle = state[0]
    return (angle > params["alpha"] + params["gamma"]) or (angle < -params["alpha"] + params["gamma"])


def calculate_state_after_event(t, state, params):
    angle = state[0]
    ang_vel = state[1]

    new_ang_vel = ang_vel * np.cos(2 * params["alpha"])
    if angle > params["alpha"] + params["gamma"]:
        new_angle = angle - 2 * params["alpha"]
        return np.array([new_angle, new_ang_vel])
    else:
        new_angle = angle + 2 * params["alpha"]
        return np.array([new_angle, new_ang_vel])