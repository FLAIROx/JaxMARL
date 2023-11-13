# Simple GridWorld in JAX

This is a simple procedurally-generated GridWorld in JAX by @minqi, which mostly follows the gymnax interface. It is largely modeled after the MiniGrid environment. See `interactive.py` for an example of how to construct an environment instance, as well as manually control the maze navigation agent with your keyboard (useful for debugging). 

Try running it as `python -m gridworld.interactive` from within the `multiagentgymnax` folder. You can include the optional flag, `--render_agent_view` to also visualize the agent's egocentric observation in a separate window.

## Code use
Note much of this code is taken from another in-progress project that will likely be open-sourced sometime over the summer, so it would be great to not open source the Maze environment _as is_, but a multi-agent extension of it would be fantastic (and should be included in jaxmarl)!

The main motivation for including this chunk of code is to provide a working example of a procedurally-generated maze that can be used as a basis for other gridworld environments in jaxmarl. 

Currently, we have plans to create
- Overcooked with partial observability (Andrei)
- Cultural Learning GridWorlds (Jonny)