from .grid_map import GridMapCircleAgents, GridMapPolygonAgents, GridMapBarn, GridMapPolygonAgentsSingleMap, GridMapFromBuffer
from .map import Map

def make_map(map_id: str, num_agents: int, rad: float, **map_kwargs) -> GridMapCircleAgents:  # note this type hint technically should be Map

    if map_id not in registered_maps:
        raise ValueError(f"Map: {map_id} not registered!")
    if map_id == "Grid-Rand":
        return GridMapCircleAgents(num_agents=num_agents, rad=rad, **map_kwargs)
    if map_id == "Grid-Rand-Poly":
        return GridMapPolygonAgents(num_agents=num_agents, rad=rad, **map_kwargs)
    if map_id == "Grid-Rand-Barn":
        return GridMapBarn(num_agents=num_agents, rad=rad, **map_kwargs)
    if map_id == "Grid-Rand-Poly-Single":
        return GridMapPolygonAgentsSingleMap(num_agents=num_agents, rad=rad, **map_kwargs)
    if map_id == "Grid-Buffer":
        return GridMapFromBuffer(num_agents=num_agents, rad=rad, **map_kwargs)

registered_maps = [
    "Grid-Rand",
    "Grid-Rand-Poly",
    "Grid-Rand-Barn",
    "Grid-Rand-Poly-Single",
    "Grid-Buffer",
]