class MemoryStream:
    def __init__(self):
        self.memories = []
        self.embeddings = []

    """"
    Input: 
    experience - str, natural language description of an experience
    timestep - int, current timestep to condition reflective experiences
    """
    def record_experience(self, experience, timestep):
        self.memories.append((experience, timestep))

        #add reflective memory every 2 timesteps
        if timestep % 2 == 0:
            self.reflect()

    def retrieve_memories(self, query):
        return self.memories

    def reflect(self):
        # Reflect on memories and draw conclusions
        print("Reflecting on memories...")
        for memory, timestep in self.memories:
            print(f"At timestep {timestep}: {memory}")