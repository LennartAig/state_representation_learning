import argparse
from srl_framework.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    # experiment file
    parser.add_argument(
        "--experiment_file", default="experiment_sac.yaml", type=str
    )
    # environment
    parser.add_argument("--env_type", default="dmc")
    parser.add_argument("--domain_name", default="cartpole")
    parser.add_argument("--task_name", default="swingup")
    parser.add_argument("--obs_type", default="state")
    parser.add_argument("--normalize_obs", default=False)
    parser.add_argument("--input_image_size", default=84, type=int)
    parser.add_argument("--action_repeat", default=4, type=int)
    parser.add_argument("--frame_stack", default=1, type=int)
    parser.add_argument("--render_image_size", default=100, type=int)
    parser.add_argument("--cameras", default=[8, 10], type=int)
    parser.add_argument("--reward_type", default="dense")
    # trainig params
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--steps_per_epoch", default=2048, type=int)
    parser.add_argument("--test_episodes", default=10, type=int)
    parser.add_argument("--initial_update_steps", default=0, type=int)
    parser.add_argument("--buffer_size", default=2048, type=int)
    # seed
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--video_freq", default=25, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()
