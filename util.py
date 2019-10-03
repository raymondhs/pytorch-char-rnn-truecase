def copy_state(state, do_clone=True):
    if do_clone:
        return (state[0].clone().detach(), state[1].clone().detach())
    else:
        return (state[0].detach(), state[1].detach())
