from .algos.ppo.ppo import PPOAgent
from .algos.ppo.ppo2 import PPOAgent2
#from .algos.sac.sac import SACAgent
from .algos.sac.sac import SACAgent
from .algos.trpo.trpo import TRPOAgent

def get_rl_agent_by_name(name):
    rl_agents = {
        'sac': [SACAgent, 'off_policy'],
        'ppo': [PPOAgent, 'on_policy'],
        'ppo2': [PPOAgent2, 'on_policy'],
        'trpo': [TRPOAgent, 'on_policy'],

    }
    if name in rl_agents:
        return rl_agents[name]
    else:
        raise 'The given agent type is not known'