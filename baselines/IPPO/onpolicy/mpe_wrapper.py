from smax.wrappers import ArrayInterface


class MPEWrapper(ArrayInterface):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = None