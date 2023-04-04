import jax 

import jax.numpy as jnp 

@jax.jit
def test_func():
    x = jnp.arange(4)

    print(jnp.where(x!=3))
    
test_func()
