from locust import HttpUser, task, events, between
from locust.runners import STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP, WorkerRunner
from locust import LoadTestShape
from locust.contrib.fasthttp import FastHttpUser
import random
import os
import base64
import time
import gevent


# Checker function to stop if the fail ratio is more than 1% of incoming requests
def checker(environment):
    while not environment.runner.state in [
        STATE_STOPPING,
        STATE_STOPPED,
        STATE_CLEANUP,
    ]:
        time.sleep(1)
        if environment.runner.stats.total.fail_ratio > 0.01:
            print(
                f"fail ratio was {environment.runner.stats.total.fail_ratio}, quitting"
            )
            environment.runner.quit()
            return


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    print("Spawning checker.")
    gevent.spawn(checker, environment)


# Create a HTTP User class to mimic end user behavior
class QuickstartUser(FastHttpUser):
    wait_time = between(0.25, 0.5)

    # @task specifies the task performed by the end user
    # picks a random image from input directory and invokes the API Gateway
    @task
    def predict(self):
        with open(f"plants.jpg", "rb") as f:
            image = f.read()

        payload = {"data": str(image), "endpoint": os.environ["endpoint"]}

        start_time = time.time()
        with self.client.post("", json=payload) as response:

            if type(response.json()) == dict:

                if response.json().get("body"):
                    total_time = int((time.time() - start_time) * 1000)
                    events.request_failure.fire(
                        request_type="POST /dev/imageclassifier",
                        name="MODELERROR",
                        response_time=total_time,
                        exception=response.json().get("body"),
                        response_length=0,
                    )
