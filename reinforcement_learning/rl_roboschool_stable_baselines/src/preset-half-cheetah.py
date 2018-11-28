import argparse

from sagemaker_rl.stable_baselines_launcher import SagemakerStableBaselinesPPO1Launcher, create_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default="/opt/ml/output/intermediate/", type=str)
    parser.add_argument('--num_timesteps', default=1e4)
    parser.add_argument('--timesteps_per_actorbatch', default=2048, type=int)
    parser.add_argument('--clip_param', default=0.2, type=float)
    parser.add_argument('--entcoeff', default=0.0, type=float)
    parser.add_argument('--optim_epochs', default=10, type=int)
    parser.add_argument('--optim_stepsize', default=3e-4)
    parser.add_argument('--optim_batchsize', default=64, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lam', default=0.95, type=float)
    parser.add_argument('--schedule', default="linear", type=str)
    parser.add_argument('--verbose', default=1, type=int)

    return parser.parse_known_args()


def main():
    args, unknown = parse_args()

    print("Launching training script with stable baselines PPO1")
    SagemakerStableBaselinesPPO1Launcher(
        env=create_env(env_id="RoboschoolHalfCheetah-v1", output_path=args.output_path),
        output_path=args.output_path,
        timesteps_per_actorbatch=int(args.timesteps_per_actorbatch),
        clip_param=float(args.clip_param),
        entcoeff=float(args.entcoeff),
        optim_epochs=int(args.optim_epochs),
        optim_stepsize=float(args.optim_stepsize),
        optim_batchsize=int(args.optim_batchsize),
        gamma=args.gamma,
        lam=float(args.lam),
        schedule=args.schedule,
        verbose=int(args.verbose),
        num_timesteps=float(args.num_timesteps)).run()


if __name__ == "__main__":
    print("Starting training for half cheetah with PPO1")
    main()
