def take_step(t, state, params, model, timestep):
    final_state = state + timestep * model.dynamics(t, state, params)
    return final_state