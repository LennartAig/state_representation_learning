import gym#, gym_duckietown, gym_miniworld, mazeexplorer
import srl_framework.envs.dmc_wrapper as dmc2gym2
import dmc2gym
from srl_framework.envs.gym_wrapper import GymWrapper, FilterObservation
from srl_framework.envs.wrappers import FrameStack, NormalizeWrapper, ResizeWrapper, PyTorchObsWrapper, NormalizeWrapper255
from srl_framework.envs.robot_wrapper import make

ROBOT_ENVS = ['FetchPickAndPlace-v1','FetchPush-v1','FetchReach-v1', 'FetchSlide-v1']

# Adapt height and width to render if contrastive loss is used:
def make_env(args, image_size):
    # Set environment
    from_pixels = args.obs_type == 'pixels'
    if args.env_type == 'dmc':
        if args.domain_name == 'manipulation':
            env = dmc2gym2.make(domain_name=args.domain_name,
                                    task_name=args.task_name,
                                    visualize_reward=False,
                                    from_pixels=from_pixels,
                                    height=image_size,
                                    width=image_size,
                                    seed = args.seed,
                                    frame_skip=args.action_repeat)
        else:
            camera_id = 2 if args.domain_name== 'quadruped' else 0
            env = dmc2gym.make(domain_name=args.domain_name,
                                    task_name=args.task_name,
                                    visualize_reward=False,
                                    from_pixels=from_pixels,
                                    height=image_size,
                                    width=image_size,
                                    seed = args.seed,
                                    frame_skip=args.action_repeat)

    elif args.env_type=='gym':
        if args.domain_name in ROBOT_ENVS:            
            
            env = make(
                domain_name=args.domain_name,
                task_name=None,
                seed=args.seed,
                visualize_reward=False,
                from_pixels=from_pixels,
                cameras=args.cameras,
                height=image_size,
                width=image_size,
                frame_skip=args.action_repeat,
                reward_type=args.reward_type,
                change_model=False
            )
        else:
            env = GymWrapper(gym.make(args.domain_name),
                            obs_type=args.obs_type,
                            height=image_size,
                            width=image_size,
                            action_repeat=args.action_repeat)
            env.seed(args.seed)
        env.action_space.seed(args.seed)

    if args.env_type =='ducky' or args.env_type =='procgen' or args.env_type =='maze' or args.env_type =='miniworld':
        env = ResizeWrapper(PyTorchObsWrapper(env), resize_w=image_size, resize_h=image_size)

    if args.normalize_obs and from_pixels:
        env=NormalizeWrapper(env)
    # Stack frames
    if args.frame_stack > 1 and from_pixels:
        env = FrameStack(env, args.frame_stack)
    
    # Get environment information
    if isinstance(env.action_space, gym.spaces.Box):
        envtype = "Box"
        act_dim = env.action_space.shape[0]
        act_limit = env.action_space.high[0]
        obs_shape = env.observation_space.shape
    elif isinstance(env.action_space, gym.spaces.Discrete):
        envtype = "Discrete"
        act_limit = 0
        act_dim = env.action_space.n
        obs_shape = env.observation_space.shape
    elif isinstance(env.action_space, gym.spaces.MuliVariant):
        envtype = "MultiVariant"
        act_limit = 0
        act_dim = env.action_space.n
        obs_shape = env.observation_space.shape
    else:
        raise "Environment Type not known"

    return env, envtype, obs_shape, act_dim, act_limit