class RLAgent:
    """
    Base Class for an agent.
    """
    def __init__(self):
        super(RLAgent, self).__init__()
        # FUTURE WORK: Standalone RL Agents      

    def step(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError