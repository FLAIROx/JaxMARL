# Hanabi Environment
This directory contains a MARL environment for the cooperative card game, Hanabi, implemented in JAX. It is inspired by the popular [Hanabi Learning Environment (HLE)](https://arxiv.org/pdf/1902.00506.pdf), but intended to be simpler to integrate and run with the growing ecosystem of JAX 
implemented RL research pipelines. 

This GPU accelerated Hanabi environment run with JAX implemented IPPO can sample and train on *10 billion* environment steps of 2-player Hanabi in *< 2 hours* when using an MLP and *< 7 hours* when using an RNN. The environment also supports custom configurations for game settings such as the 
number of colours/ ranks, number of players, number of hint tokens, etc.
