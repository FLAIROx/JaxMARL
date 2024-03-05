# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from obl.net import FFWDNet, PublicLSTMNet, LSTMNet
import numpy as np


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = [
        "vdn",
        "multi_step",
        "gamma",
        "eta",
        "boltzmann",
        "uniform_priority",
        "net",
    ]

    def __init__(
        self,
        vdn,
        multi_step,
        gamma,
        eta,
        device,
        in_dim,
        hid_dim,
        out_dim,
        net,
        num_lstm_layer,
        boltzmann_act,
        uniform_priority,
        off_belief,
        weight_file,
        greedy=False,
        nhead=None,
        nlayer=None,
        max_len=None,
        parameterized=0,
        parameter_type="one_hot",
        num_parameters=0,
        num_partners=0,
    ):
        super().__init__()

        if net == "ffwd":
            self.online_net = FFWDNet(in_dim, hid_dim, out_dim).to(device)
            self.target_net = FFWDNet(in_dim, hid_dim, out_dim).to(device)
        elif net == "publ-lstm":
            self.online_net = PublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
            self.target_net = PublicLSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer
            ).to(device)
        elif net == "lstm":
            self.online_net = LSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer, num_partners
            ).to(device)
            self.target_net = LSTMNet(
                device, in_dim, hid_dim, out_dim, num_lstm_layer, num_partners
            ).to(device)
        elif net == "transformer":
            self.online_net = TransformerNet(
                device, in_dim, hid_dim, out_dim, nhead, nlayer, max_len
            )
            self.target_net = TransformerNet(
                device, in_dim, hid_dim, out_dim, nhead, nlayer, max_len
            )
        else:
            assert False, f"{net} not implemented"

        for p in self.target_net.parameters():
            p.requires_grad = False

        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.net = net
        self.num_lstm_layer = num_lstm_layer
        self.boltzmann = boltzmann_act
        self.uniform_priority = uniform_priority
        self.off_belief = off_belief
        self.greedy = greedy
        self.nhead = nhead
        self.nlayer = nlayer
        self.max_len = max_len
        self.device = device
        self.parameterized = parameterized
        self.parameter_type = parameter_type
        self.num_parameters = num_parameters
        self.model_name = weight_file
        self.num_partners = num_partners

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    @torch.jit.script_method
    def get_model_name(self) -> str:
        return self.model_name

    @torch.jit.script_method
    def get_model_device(self) -> str:
        return self.device


    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.net,
            self.num_lstm_layer,
            overwrite.get("boltzmann_act", self.boltzmann),
            self.uniform_priority,
            self.off_belief,
            self.model_name,
            self.greedy,
            nhead=self.nhead,
            nlayer=self.nlayer,
            max_len=self.max_len,
            parameterized=self.parameterized,
            parameter_type=self.parameter_type,
            num_parameters=self.num_parameters,
            num_partners=self.num_partners,
        )
        cloned.load_state_dict(self.state_dict())
        cloned.train(self.training)
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        adv, new_hid = self.online_net.act(priv_s, publ_s, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid, legal_adv

    @torch.jit.script_method
    def boltzmann_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        temperature: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        temperature = temperature.unsqueeze(1)
        adv, new_hid = self.online_net.act(priv_s, publ_s, hid)
        assert adv.dim() == temperature.dim()
        logit = adv / temperature
        legal_logit = logit - (1 - legal_move) * 1e30
        assert legal_logit.dim() == 2
        prob = nn.functional.softmax(legal_logit, 1)
        action = prob.multinomial(1).squeeze(1).detach()
        return action, new_hid, prob

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        priv_s = obs["priv_s"]
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]

        if "eps" in obs:
            eps = obs["eps"].flatten(0, 1)
        else:
            eps = torch.zeros((priv_s.size(0),), device=priv_s.device)

        if self.vdn:
            bsize, num_player = obs["priv_s"].size()[:2]
            priv_s = obs["priv_s"].flatten(0, 1)
            publ_s = obs["publ_s"].flatten(0, 1)
            legal_move = obs["legal_move"].flatten(0, 1)
        else:
            bsize, num_player = obs["priv_s"].size()[0], 1

        hid = {"h0": obs["h0"], "c0": obs["c0"]}

        if self.boltzmann:
            temp = obs["temperature"].flatten(0, 1)
            greedy_action, new_hid, prob = self.boltzmann_act(
                priv_s, publ_s, legal_move, temp, hid
            )
            reply = {"prob": prob}
            legal_act = prob
        else:
            greedy_action, new_hid, legal_act = \
                    self.greedy_act(priv_s, publ_s, legal_move, hid)
            reply = {}

        random_action = legal_move.multinomial(1).squeeze(1)

        rand = torch.rand(greedy_action.size(), device=greedy_action.device, dtype=torch.double)

        if self.greedy:
            taking_exp_action = torch.tensor([0])
            action = greedy_action
        else: 
            assert rand.size() == eps.size()
            taking_exp_action = rand < eps
            action = torch.where(taking_exp_action, random_action, greedy_action).detach()

        if self.vdn:
            action = action.view(bsize, num_player)
            greedy_action = greedy_action.view(bsize, num_player)
            rand = rand.view(bsize, num_player)

        reply["a"] = action.detach().cpu()
        reply["h0"] = new_hid["h0"].detach().cpu()
        reply["c0"] = new_hid["c0"].detach().cpu()
        reply["all_q"] = legal_act.detach().cpu()
        reply["legal_moves"] = legal_move.detach().cpu()
        reply["explore_a"] = taking_exp_action.detach().cpu()

        size = torch.prod(torch.tensor(legal_move.shape))
        ascending = torch.arange(0, size, legal_move.shape[1])
        take_indices = reply["a"] + ascending
        selected_action = torch.take(legal_move.detach().cpu(), take_indices)

        # Check for illegal action, and raise exception if found.
        if torch.any(selected_action == 0) == 1:
            raise Exception("Found illegal action")

        return reply

    @torch.jit.script_method
    def compute_target(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        assert self.multi_step == 1
        priv_s = input_["priv_s"]
        publ_s = input_["publ_s"]
        legal_move = input_["legal_move"]
        act_hid = {
            "h0": input_["h0"],
            "c0": input_["c0"],
        }
        fwd_hid = {
            "h0": input_["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": input_["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }
        reward = input_["reward"]
        terminal = input_["terminal"]
        convention_idx = input_["convention_idx"]

        priv_s, publ_s = self.encode_parameters(priv_s, publ_s, convention_idx)

        if self.boltzmann:
            temp = input_["temperature"].flatten(0, 1)
            next_a, _, next_pa = self.boltzmann_act(
                priv_s, publ_s, legal_move, temp, act_hid
            )
            next_q = self.target_net(priv_s, publ_s, legal_move, next_a, fwd_hid)[2]
            qa = (next_q * next_pa).sum(1)
        else:
            next_a = self.greedy_act(priv_s, publ_s, legal_move, act_hid)[0]
            qa = self.target_net(priv_s, publ_s, legal_move, next_a, fwd_hid)[0]

        assert reward.size() == qa.size()
        target = reward + (1 - terminal) * self.gamma * qa
        return {"target": target.detach()}

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.uniform_priority:
            return {"priority": torch.ones_like(input_["reward"].sum(1))}

        # swap batch_dim and seq_dim
        for k, v in input_.items():
            if k != "seq_len":
                input_[k] = v.transpose(0, 1).contiguous()

        obs = {
            "priv_s": input_["priv_s"],
            "publ_s": input_["publ_s"],
            "legal_move": input_["legal_move"],
            "convention_idx": input_["convention_idx"]
        }
        if self.boltzmann:
            obs["temperature"] = input_["temperature"]

        if self.off_belief:
            obs["target"] = input_["target"]

        hid = {"h0": input_["h0"], "c0": input_["c0"]}
        action = {"a": input_["a"]}
        reward = input_["reward"]
        terminal = input_["terminal"]
        bootstrap = input_["bootstrap"]
        seq_len = input_["seq_len"]
        err, _, _ = self.td_error(
            obs, hid, action, reward, terminal, bootstrap, seq_len
        )
        priority = err.abs()
        priority = self.aggregate_priority(priority, seq_len).detach().cpu()
        return {"priority": priority}

    @torch.jit.script_method
    def td_error(
        self,
        obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_seq_len = obs["priv_s"].size(0)
        priv_s = obs["priv_s"]
        publ_s = obs["publ_s"]
        legal_move = obs["legal_move"]
        action = action["a"]
        convention_idx = obs["convention_idx"]

        priv_s, publ_s = self.encode_parameters(priv_s, publ_s, convention_idx)

        for k, v in hid.items():
            hid[k] = v.flatten(1, 2).contiguous()

        bsize, num_player = priv_s.size(1), 1
        if self.vdn:
            num_player = priv_s.size(2)
            priv_s = priv_s.flatten(1, 2)
            publ_s = publ_s.flatten(1, 2)
            legal_move = legal_move.flatten(1, 2)
            action = action.flatten(1, 2)

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        online_qa, greedy_a, online_q, lstm_o = self.online_net(
            priv_s, publ_s, legal_move, action, hid
        )

        if self.off_belief:
            target = obs["target"]
        else:
            target_qa, _, target_q, _ = self.target_net(
                priv_s, publ_s, legal_move, greedy_a, hid
            )

            if self.boltzmann:
                temperature = obs["temperature"].flatten(1, 2).unsqueeze(2)
                # online_q: [seq_len, bathc * num_player, num_action]
                logit = online_q / temperature.clamp(min=1e-6)
                # logit: [seq_len, batch * num_player, num_action]
                legal_logit = logit - (1 - legal_move) * 1e30
                assert legal_logit.dim() == 3
                pa = nn.functional.softmax(legal_logit, 2).detach()
                # pa: [seq_len, batch * num_player, num_action]

                assert target_q.size() == pa.size()
                target_qa = (pa * target_q).sum(-1).detach()
                assert online_qa.size() == target_qa.size()

            if self.vdn:
                online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
                target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)
                lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)

            target_qa = torch.cat(
                [target_qa[self.multi_step :], target_qa[: self.multi_step]], 0
            )
            target_qa[-self.multi_step :] = 0
            assert target_qa.size() == reward.size()
            target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa

        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        err = (target.detach() - online_qa) * mask
        if self.off_belief and "valid_fict" in obs:
            err = err * obs["valid_fict"]
        return err, lstm_o, online_q

    def to_array(self, tensor):
        return np.array(tensor.clone().detach().cpu())

    def class_aux_task_iql(self, lstm_o, partner_idx, num_partners, 
            seq_len, rl_loss_size, stat):
        num_partners = num_partners.cpu()[0,0].item()
        one_hot = F.one_hot(partner_idx, 
                num_classes=num_partners)
        one_hot = torch.unsqueeze(one_hot, 2)
        mask = one_hot.sum(3)

        pred_loss1, avg_xent1, probs, _ = self.online_net.pred_loss_class(
            lstm_o, one_hot, partner_idx, mask, seq_len
        )

        selection = self.to_array(torch.argmax(probs, -1)).squeeze()
        equal = selection == self.to_array(partner_idx)
        equal_all_summed = equal.sum()
        total_all = np.prod([*selection.shape])
        aux_accuracy = equal_all_summed / total_all

        aux_accuracy_per_step = equal.sum(axis=1) / selection.shape[1]
        
        assert pred_loss1.size() == rl_loss_size
        stat["aux"].feed(avg_xent1)

        return pred_loss1, aux_accuracy, list(aux_accuracy_per_step)


    def aux_task_iql(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        seq_size, bsize, _ = hand.size()

        own_hand = hand.view(seq_size, bsize, 5, 3)
        own_hand_slot_mask = own_hand.sum(3)

        pred_loss1, avg_xent1, _, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size
        stat["aux"].feed(avg_xent1)
        return pred_loss1

    def aux_task_vdn(self, lstm_o, hand, t, seq_len, rl_loss_size, stat):
        """1st and 2nd order aux task used in VDN"""
        seq_size, bsize, num_player, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, num_player, 5, 3)
        own_hand_slot_mask = own_hand.sum(4)
        pred_loss1, avg_xent1, belief1, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        stat["aux"].feed(avg_xent1)
        return pred_loss1

    def aggregate_priority(self, priority, seq_len):
        p_mean = priority.sum(0) / seq_len
        p_max = priority.max(0)[0]
        agg_priority = self.eta * p_max + (1.0 - self.eta) * p_mean
        return agg_priority

    def loss(self, batch, aux_weight, stat, class_aux_weight):
        err, lstm_o, online_q = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())

        priority = err.abs()
        priority = self.aggregate_priority(priority, batch.seq_len).detach().cpu()

        loss = rl_loss
        aux_loss = 0
        aux_accuracy = 0
        aux_accuracies = []
    
        if aux_weight > 0:
            if self.vdn:
                pred1 = self.aux_task_vdn(
                    lstm_o,
                    batch.obs["own_hand"],
                    batch.obs["temperature"],
                    batch.seq_len,
                    rl_loss.size(),
                    stat,
                )
                loss = rl_loss + aux_weight * pred1
            else:
                pred = self.aux_task_iql(
                    lstm_o,
                    batch.obs["own_hand"],
                    batch.seq_len,
                    rl_loss.size(),
                    stat,
                )
                loss = rl_loss + aux_weight * pred

        if class_aux_weight > 0:
            pred, aux_accuracy, aux_accuracies = self.class_aux_task_iql(
                lstm_o,
                batch.obs["partner_idx"],
                batch.obs["num_partners"],
                batch.seq_len,
                rl_loss.size(),
                stat,
            )
            aux_loss = pred
            loss = rl_loss + class_aux_weight * pred

        return loss, priority, online_q, aux_loss, aux_accuracy, aux_accuracies

    def behavior_clone_loss(self, online_q, batch, t, clone_bot, stat):
        max_seq_len = batch.obs["priv_s"].size(0)
        priv_s = batch.obs["priv_s"]
        publ_s = batch.obs["publ_s"]
        legal_move = batch.obs["legal_move"]

        bsize, num_player = priv_s.size(1), 1
        if self.vdn:
            num_player = priv_s.size(2)
            priv_s = priv_s.flatten(1, 2)
            publ_s = publ_s.flatten(1, 2)
            legal_move = legal_move.flatten(1, 2)

        with torch.no_grad():
            target_logit, _ = clone_bot(priv_s, publ_s, None)
            target_logit = target_logit - (1 - legal_move) * 1e10
            target = nn.functional.softmax(target_logit, 2)

        logit = online_q / t
        # logit: [seq_len, batch * num_player, num_action]
        legal_logit = logit - (1 - legal_move) * 1e10
        log_distq = nn.functional.log_softmax(legal_logit, 2)

        assert log_distq.size() == target.size()
        assert log_distq.size() == legal_move.size()
        xent = -(target.detach() * log_distq).sum(2) / legal_move.sum(2).clamp(min=1e-3)
        if self.vdn:
            xent = xent.view(max_seq_len, bsize, num_player).sum(2)

        mask = torch.arange(0, max_seq_len, device=batch.seq_len.device)
        mask = (mask.unsqueeze(1) < batch.seq_len.unsqueeze(0)).float()
        assert xent.size() == mask.size()
        xent = xent * mask
        xent = xent.sum(0)
        stat["bc_loss"].feed(xent.mean().detach())
        return xent

    def encode_parameters(self, priv_s, publ_s, convention_idx):
        if not self.parameterized or self.num_parameters == 0:
            return priv_s, publ_s

        if self.parameter_type == "one_hot":
            one_hot = F.one_hot(convention_idx, 
                    num_classes=self.num_parameters)
            priv_s = torch.cat((priv_s, one_hot), len(priv_s.shape) - 1)
            publ_s = torch.cat((publ_s, one_hot), len(publ_s.shape) - 1)
        elif self.parameter_type == "two_hot":
            num_colours = 5
            num_ranks = 5
            colour_idx = torch.div(convention_idx, num_colours,
                    rounding_mode="trunc")
            rank_idx = torch.remainder(convention_idx, num_colours)
            colour_one_hot = F.one_hot(colour_idx, num_classes=num_colours)
            rank_one_hot = F.one_hot(rank_idx, num_classes=num_ranks)
            cat_axis = len(priv_s.shape) - 1
            priv_s = torch.cat((priv_s, colour_one_hot, rank_one_hot), cat_axis)
            publ_s = torch.cat((publ_s, colour_one_hot, rank_one_hot), cat_axis)

        return priv_s, publ_s
