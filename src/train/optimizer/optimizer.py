import optax

# Basic optimizer 
def get_optimizer(learning_rate, grad_clipping):
    opt =  optax.chain(
        optax.clip_by_global_norm(grad_clipping),  # Gradient clipping
        optax.adam(learning_rate=learning_rate)    # Optimizer of choice, e.g., Adam
    )
    return opt