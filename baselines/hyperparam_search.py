import itertools
from collections import defaultdict
import jax.numpy as jnp
import dataclasses

def wrap_singletons_in_list(mapping):
    ret = {}
    for k, v in mapping.items():
        ret[k] = v if isinstance(v, list) else [v]
    return ret


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def list_of_dicts_to_pytree(configs, struct_keys):
    pytree_dict = defaultdict(list)
    for config in configs:
        for k, v in config.items():
            pytree_dict[k].append(v)

    ret = {
        k: jnp.array(v) if k not in struct_keys else list_of_structs_to_array(v)
        for k, v in pytree_dict.items()
    }
    return ret


def list_of_structs_to_array(struct_list):
    cls = type(struct_list[0])
    return cls(
        **{k.name: jnp.array([getattr(v, k.name) for v in struct_list]) for k in dataclasses.fields(cls)}
    )


def generate_vmappable_config(config, struct_keys=set(), preprocess_fns={}):
    """Generates a vmappable config from a config which is a dict of lists.
    Allows the transformation of some keys in your config using `preprocess_fns`.
    This is useful for example if you want to vmap over SMAX maps, which are specified
    by a string. You can then call `map_name_to_scenario` by passing `{"MAP_NAME": map_name_to_scenario}`
    into `preprocess_fns`.

    This also requires any keys that are structs (as opposed to simple types like `int`, `float` etc.)
    to be specified in `struct_keys`. This also helps to handle SMAX maps for example, which return a
    `flax.struct.dataclass` from `map_name_to_scenario`. To vmap over this requires a dataclass of arrays, not
    a list of dataclasses.

    Args:
        config (dict): A dict of lists from which to generate the vmappable hyperparam config
        struct_keys (set): The set of keys which are structs and hence must be processed differently
        preprocess_fns (dict): A mapping from key in the config to preprocessing function to perform
    """
    config = wrap_singletons_in_list(config)
    config = list(product_dict(**config))
    preprocessed_config = []
    for cfg in config:
        new_cfg = {}
        for k, v in cfg.items():
            new_cfg[k] = v if k not in preprocess_fns else preprocess_fns[k](v)
        preprocessed_config.append(new_cfg)
    return list_of_dicts_to_pytree(preprocessed_config, struct_keys)



if __name__ == "__main__":
    from smax.environments.mini_smac import map_name_to_scenario
    config = {"a": [1, 2, 3], "b": [5, 6], "c": 1, "d": ["smacv2_5_units", "2s3z"]}

    vmap_config = generate_vmappable_config(config, struct_keys={"d"}, preprocess_fns={"d": map_name_to_scenario})
    print(vmap_config)