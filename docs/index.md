# Welcome to JaxMARL!


<div class="collage">
    <div class="column" align="centre">
        <div class="row" align="centre">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/cramped_room.gif?raw=true" alt="Overcooked" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/mabrax.png?raw=true" alt="mabrax" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/storm.gif?raw=true" alt="STORM" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/hanabi.png?raw=true" alt="hanabi" width="20%">
        </div>
        <div class="row" align="centre">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/coin_game.png?raw=true" alt="coin_game" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/qmix_MPE_simple_tag_v3.gif?raw=true" alt="MPE" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/jaxnav-ma.gif?raw=true" alt="jaxnav" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/smax.gif?raw=true" alt="SMAX" width="20%">
        </div>
    </div>
</div>

_**MARL but really really fast!**_

JaxMARL combines ease-of-use with GPU-enabled efficiency, and supports a wide range of commonly used MARL environments as well as popular baseline algorithms. Our aim is for one library that enables thorough evaluation of MARL methods across a wide range of tasks and against relevant baselines. We also introduce SMAX, a vectorised, simplified version of the popular StarCraft Multi-Agent Challenge, which removes the need to run the StarCraft II game engine. 

## What we provide:
* **9 MARL environments** fully implemented in JAX - these span cooperative, competitive, and mixed games; discrete and continuous state and action spaces; and zero-shot and CTDE settings.
* **8 MARL algorithms**, also fully implemented in JAX - these include both Q-Learning and PPO based appraoches.

## Who is JaxMARL for?
Anyone doing research on or looking to use multi-agent reinforcment learning!

## What is JAX?

[JAX](https://jax.readthedocs.io/en/latest/) is a Python library that enables programmers to use a simple numpy-like interface to easily run programs on accelerators. Recently, doing end-to-end single-agent RL on the accelerator using JAX has shown incredible benefits. To understand the reasons for such massive speed-ups in depth, we recommend reading the [PureJaxRL blog post](https://chrislu.page/blog/meta-disco/) and [repository](https://github.com/luchris429/purejaxrl).

## Performance Examples
*coming soon*

## Related Works
This works is heavily related to and builds on many other works. We would like to highlight some of the works that we believe would be relevant to readers:

* [Jumanji](https://github.com/instadeepai/jumanji). A suite of JAX-based RL environments. It includes some multi-agent ones such as RobotWarehouse.
* [VectorizedMultiAgentSimulator (VMAS)](https://github.com/proroklab/VectorizedMultiAgentSimulator). It performs similar vectorization for some MARL environments, but is done in PyTorch.
* More to be added soon :)

More documentation to follow soon!

## Citing JaxMARL
If you use JaxMARL in your work, please cite us as follows:
```bibtex
@article{flair2023jaxmarl,
    title={JaxMARL: Multi-Agent RL Environments in JAX},
    author={Alexander Rutherford and Benjamin Ellis and Matteo Gallici and Jonathan Cook and Andrei Lupu and Gardar Ingvarsson and Timon Willi and Akbir Khan and Christian Schroeder de Witt and Alexandra Souly and Saptarashmi Bandyopadhyay and Mikayel Samvelyan and Minqi Jiang and Robert Tjarko Lange and Shimon Whiteson and Bruno Lacerda and Nick Hawes and Tim Rocktaschel and Chris Lu and Jakob Nicolaus Foerster},
    journal={arXiv preprint arXiv:2311.10090},
    year={2023}
}
```