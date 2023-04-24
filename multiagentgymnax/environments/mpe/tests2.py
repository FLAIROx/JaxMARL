import jax 

import jax.numpy as jnp 


def test_func(x):
    print('-- x', x)
    
    
d = {i: jnp.array([i, i**2])  for i in range(5)}

t = jax.vmap(test_func, in_axes=(0,))
print(d)
print(t(d))