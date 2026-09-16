import numpy as np

# 4th order Runge-Kutta method
def take_step(t, state, params, model, timestep):
    state_derivatives = np.empty(4, dtype=object)
    state_derivatives[0] = model.dynamics(
        t,
        state,
        params
    )
    state_derivatives[1] = model.dynamics(
        t + timestep/2,
        state + state_derivatives[0] * timestep/2,
        params
    )
    state_derivatives[2] = model.dynamics(
        t + timestep/2,
        state + state_derivatives[1] * timestep/2,
        params
    )
    state_derivatives[3] = model.dynamics(
        t + timestep,
        state + state_derivatives[2] * timestep,
        params
    )
    avg_state_derivative = np.array(
        (1/6) * (state_derivatives[0] +
        2*state_derivatives[1] +
        2*state_derivatives[2] +
        state_derivatives[3])
        )
    return state + timestep * avg_state_derivative