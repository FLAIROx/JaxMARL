import torch
import os
import sys
import random
import json
from collections import OrderedDict
import numpy as np
import jax
import jax.numpy as jnp

import obl.r2d2 as r2d2

class OBLPytorchAgent:
    def __init__(self, weight_file, device="cuda:0"):
        self._agent = load_agent_from_file(weight_file, device)
        self._device = device
        self._hid = {
            'h0': torch.zeros(1, 2, 2, 512).to(device),
            'c0':torch.zeros(1, 2, 2, 512).to(device),
        }

    def act(self, obs, legal_moves, curr_player):
        obs = self._batchify(obs)
        legal_moves = self._batchify(legal_moves)
        #legal_moves = jnp.roll(legal_moves, -1, axis=1)

        torch_obs = {
            'priv_s':torch.tensor(np.array(obs[..., 125:])).to(self._device),
            'publ_s':torch.tensor(np.array(obs[..., 250:])).to(self._device),
            'h0': self._hid['h0'].to(self._device),
            'c0': self._hid['c0'].to(self._device),
            'legal_move': torch.tensor(np.array(legal_moves)).to(self._device),
        }
        
        act_result = self._agent.act(torch_obs)
        actions = act_result.pop('a').detach().numpy()
        self._hid = act_result
        #actions = np.where(actions+1==21, 0, actions+1)
        return actions

    def _batchify(self, x_dict): 
        return jnp.stack([x_dict[agent] for agent in sorted(x_dict)])


def load_agent_from_file(weight_file, device, sad_legacy=False, iql_legacy=False):
    # Load models from weight files
    assert os.path.exists(weight_file), f"path file not found: {weight_file}"

    try: 
        torch.load(weight_file)
    except:
        sys.exit(f"weight_file {weight_file} can't be loaded")

    overwrite = {'vdn': False, 'device': 'cuda:0', 'boltzmann_act': False}
    agent, _ = load_agent(weight_file, overwrite)

    agent.train(False)

    return agent


def load_agent(weight_file, overwrite):
    """
    overwrite has to contain "device"
    """
    print("loading file from: ", weight_file)
    cfg = get_train_config(weight_file)
    assert cfg is not None

    if "core" in cfg:
        new_cfg = {}
        flatten_dict(cfg, new_cfg)
        cfg = new_cfg

    cfg["parameterized"] = cfg["parameterized"] if "parameterized" in cfg else False
    cfg["parameter_type"] = cfg["parameter_type"] if "parameter_type" in cfg else "one_hot"
    cfg["num_parameters"] = cfg["num_parameters"] if "num_parameters" in cfg else 0

    in_dim = (783, 658, 533)

    if cfg["parameterized"]:
        in_dim = tuple([x + cfg["num_parameters"] for x in in_dim])

    config = {
        "vdn": overwrite["vdn"] if "vdn" in overwrite else cfg["method"] == "vdn",
        "multi_step": overwrite.get("multi_step", cfg["multi_step"]),
        "gamma": overwrite.get("gamma", cfg["gamma"]),
        "eta": 0.9,
        "device": overwrite["device"],
        "in_dim": in_dim,
        "hid_dim": cfg["hid_dim"] if "hid_dim" in cfg else cfg["rnn_hid_dim"],
        "out_dim": 21,
        "num_lstm_layer": cfg.get("num_lstm_layer", overwrite.get("num_lstm_layer", 2)),
        "boltzmann_act": overwrite.get(
            "boltzmann_act", cfg.get("boltzmann_act", False)
        ),
        "uniform_priority": overwrite.get("uniform_priority", False),
        "net": cfg.get("net", "publ-lstm"),
        "off_belief": overwrite.get("off_belief", cfg.get("off_belief", False)),
        "parameterized": cfg["parameterized"],
        "parameter_type": cfg["parameter_type"],
        "num_parameters": cfg["num_parameters"],
        "weight_file": weight_file,
    }
    if cfg.get("net", None) == "transformer":
        config["nhead"] = cfg["nhead"]
        config["nlayer"] = cfg["nlayer"]
        config["max_len"] = cfg["max_len"]

    agent = r2d2.R2D2Agent(**config).to(config["device"])
    load_weight(agent.online_net, weight_file, config["device"])
    agent.sync_target_with_online()

    return agent, cfg


def get_train_config(weight_file):
    log = os.path.join(os.path.dirname(weight_file), "train.log")
    if not os.path.exists(log):
        return None

    lines = open(log, "r").readlines()
    try:
        cfg, rest = parse_first_dict(lines)
    except json.decoder.JSONDecodeError as e:
        print("ERROR =======================================")
        # cfg = parse_hydra_dict(lines)
    return cfg


def parse_first_dict(lines):
    config_lines = []
    open_count = 0
    for i, l in enumerate(lines):
        if l.strip()[0] == "{":
            open_count += 1
        if open_count:
            config_lines.append(l)
        if l.strip()[-1] == "}":
            open_count -= 1
        if open_count == 0 and len(config_lines) != 0:
            break

    config = "".join(config_lines).replace("'", '"')
    config = config.replace("True", "true")
    config = config.replace("False", "false")
    config = config.replace("None", "null")
    config = json.loads(config)
    return config, lines[i + 1 :]


def flatten_dict(d, new_dict):
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, new_dict)
        else:
            new_dict[k] = v


def load_weight(model, weight_file, device, *, state_dict=None):
    if state_dict is None:
        state_dict = torch.load(weight_file, map_location=device)

    source_state_dict = OrderedDict()
    target_state_dict = model.state_dict()

    if not set(state_dict.keys()).intersection(set(target_state_dict.keys())):
        new_state_dict = OrderedDict()
        for k in state_dict.keys():
            if "online_net" in k:
                new_k = k[len("online_net.") :]
                new_state_dict[new_k] = state_dict[k]
        state_dict = new_state_dict

    for k, v in target_state_dict.items():
        if k not in state_dict:
            # print("warning: %s not loaded [not found in file]" % k)
            state_dict[k] = v
        elif state_dict[k].size() != v.size():
            print(
                "warning: %s not loaded\n[size mismatch %s (in net) vs %s (in file)]"
                % (k, v.size(), state_dict[k].size())
            )
            state_dict[k] = v
    for k in state_dict:
        if k not in target_state_dict:
            print("removing: %s not used" % k)
        else:
            source_state_dict[k] = state_dict[k]

    model.load_state_dict(source_state_dict)
    return
