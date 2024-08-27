# 🧭 JaxNav 

2D geometric navigation for differential drive robots. Using distances readings to nearby obstacles (mimicing LiDAR readings), the direction to their goal and their current velocity, robots must navigate to their goal without colliding with obstacles.

## Environment Details

### Map Types
The default map is square robots of width 0.5m moving within a world with grid based obstacled, with cells of size 1m x 1m. Map cell size can be varied to produce obstacles of higher fidelty or robot strucutre can be changed into any polygon or a circle.

We also include a map which uses polygon obstacles, but note we have not used this code is a while so there may well be issues with it.

### Observation space
By default, each robot recieves 200 range readings from a 360-degree arc centered on their forward axis. These range readings have a max range of 6m but no minimum range and are discritised with a resultion of 0.05 m. Alongside these range readings, each robot recieves their current linear and angular velocities along with the direction to their goal. Their goal direction is given by a vector in polar form where the distance is either the max lidar range if the goal is beyond their "line of sight" or the actual distance if the goal is within their lidar range. There is no communication between agents.

### Action Space
The environments default action space is a 2D continuous action, where the first dimension is the desired linear velocity and the second the desired angular velocity. Discrete actions are also supported, where the possible combination of linear and angular velocities are discretised into 15 options.

### Reward function
By default, the reward function contains a sparse outcome based reward alongside a dense shaping term.

## Visulisation

## TODOs:
- remove self.rad dependence

## Citation
JaxNav was introduced by the following paper, if you use JaxNav in your work please cite it as:

'''bibtex
TODO
'''