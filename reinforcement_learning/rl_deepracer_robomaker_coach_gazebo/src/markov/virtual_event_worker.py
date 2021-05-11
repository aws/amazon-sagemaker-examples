""" The main loop for running virtual event """
import argparse
import logging
import os

import rospy
from markov import utils
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.virtual_event.constants import DEFAULT_RACE_DURATION
from markov.virtual_event.virtual_event_manager import VirtualEventManager

LOG = Logger(__name__, logging.INFO).get_logger()


def main():
    """Main function for virutal event manager"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue_url",
        help="the sqs queue url to receive next racer information",
        type=str,
        default=str(rospy.get_param("SQS_QUEUE_URL", "sqs_queue_url")),
    )
    parser.add_argument(
        "--race_duration",
        help="the length of the race in seconds.",
        type=int,
        default=int(rospy.get_param("RACE_DURATION", DEFAULT_RACE_DURATION)),
    )
    parser.add_argument(
        "--aws_region",
        help="(string) AWS region",
        type=str,
        default=rospy.get_param("AWS_REGION", "us-east-1"),
    )
    parser.add_argument(
        "--number_of_trials",
        help="(integer) Number of trials",
        type=int,
        default=int(rospy.get_param("NUMBER_OF_TRIALS", 3)),
    )
    parser.add_argument(
        "--number_of_resets",
        help="(integer) Number of resets",
        type=int,
        default=int(rospy.get_param("NUMBER_OF_RESETS", 0)),
    )
    parser.add_argument(
        "--penalty_seconds",
        help="(float) penalty second",
        type=float,
        default=float(rospy.get_param("PENALTY_SECONDS", 2.0)),
    )
    parser.add_argument(
        "--off_track_penalty",
        help="(float) off track penalty second",
        type=float,
        default=float(rospy.get_param("OFF_TRACK_PENALTY", 2.0)),
    )
    parser.add_argument(
        "--collision_penalty",
        help="(float) collision penalty second",
        type=float,
        default=float(rospy.get_param("COLLISION_PENALTY", 5.0)),
    )
    parser.add_argument(
        "--is_continuous",
        help="(boolean) is continous after lap completion",
        type=bool,
        default=utils.str2bool(rospy.get_param("IS_CONTINUOUS", False)),
    )
    parser.add_argument(
        "--race_type",
        help="(string) Race type",
        type=str,
        default=rospy.get_param("RACE_TYPE", "TIME_TRIAL"),
    )
    parser.add_argument(
        "--body_shell_type",
        help="(string) body shell type",
        type=str,
        default=rospy.get_param("BODY_SHELL_TYPE", "deepracer"),
    )

    args = parser.parse_args()
    manager = VirtualEventManager(
        queue_url=args.queue_url,
        aws_region=args.aws_region,
        race_duration=args.race_duration,
        number_of_trials=args.number_of_trials,
        number_of_resets=args.number_of_resets,
        penalty_seconds=args.penalty_seconds,
        off_track_penalty=args.off_track_penalty,
        collision_penalty=args.collision_penalty,
        is_continuous=args.is_continuous,
        race_type=args.race_type,
        body_shell_type=args.body_shell_type,
    )
    while True:
        # poll for next racer
        if not manager.is_event_end and manager.current_racer is None:
            LOG.info("[virtual event worker] polling for next racer.")
            manager.poll_next_racer()

        # if event end signal received, break out loop and finish the job
        if manager.is_event_end:
            LOG.info("[virtual event worker] received event end.")
            break
        # Setting up the race environment
        if manager.setup_race():
            # proceed with start and finish race only if setup is successful.
            # Start race
            manager.start_race()
            # Finish race
            manager.finish_race()

    utils.cancel_simulation_job(
        os.environ.get("AWS_ROBOMAKER_SIMULATION_JOB_ARN"), rospy.get_param("AWS_REGION")
    )


if __name__ == "__main__":
    try:
        rospy.init_node("virtual_event_manager", anonymous=True)
        main()
    except Exception as ex:
        log_and_exit(
            "Virtual event worker error: {}".format(ex),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
