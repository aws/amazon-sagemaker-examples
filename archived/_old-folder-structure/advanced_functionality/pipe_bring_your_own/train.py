#!/usr/bin/env python

import json
import os
import signal
import sys
import time

input_dir = "/opt/ml/input"
model_dir = "/opt/ml/model"
output_dir = "/opt/ml/output"

# we're arbitrarily going to iterate through 5 epochs here, a real algorithm
# may choose to determine the number of epochs based on a more realistic
# convergence criteria
num_epochs = 5
channel_name = "training"
terminated = False


def main():
    # trapping signals and responding to them appropriately is required by
    # SageMaker spec
    trap_signal()

    # writing to a failure file is also part of the spec
    failure_file = output_dir + "/failure"
    data_dir = input_dir + "/data"

    try:
        # we're allocating a byte array here to read data into, a real algo
        # may opt to prefetch the data into a memory buffer and train in
        # in parallel so that both IO and training happen simultaneously
        data = bytearray(16777216)
        total_read = 0
        total_duration = 0
        for epoch in range(num_epochs):
            check_termination()
            epoch_bytes_read = 0
            # As per SageMaker Training spec, the FIFO's path will be based on
            # the channel name and the current epoch:
            fifo_path = "{0}/{1}_{2}".format(data_dir, channel_name, epoch)

            # Usually the fifo will already exist by the time we get here, but
            # to be safe we should wait to confirm:
            wait_till_fifo_exists(fifo_path)
            with open(fifo_path, "rb", buffering=0) as fifo:
                print("opened fifo: %s" % fifo_path)
                # Now simply iterate reading from the file until EOF. Again, a
                # real algorithm will actually do something with the data
                # rather than simply reading and immediately discarding like we
                # are doing here
                start = time.time()
                bytes_read = fifo.readinto(data)
                total_read += bytes_read
                epoch_bytes_read += bytes_read
                while bytes_read > 0 and not terminated:
                    bytes_read = fifo.readinto(data)
                    total_read += bytes_read
                    epoch_bytes_read += bytes_read

                duration = time.time() - start
                total_duration += duration
                epoch_throughput = epoch_bytes_read / duration / 1000000
                print(
                    "Completed epoch %s; read %s bytes; time: %.2fs, throughput: %.2f MB/s"
                    % (epoch, epoch_bytes_read, duration, epoch_throughput)
                )

        # now write a model, again, totally meaningless contents:
        with open(model_dir + "/model.json", "w") as model:
            json.dump(
                {
                    "bytes_read": total_read,
                    "duration": total_duration,
                    "throughput_MB_per_sec": total_read / total_duration / 1000000,
                },
                model,
            )
    except Exception:
        print("Failed to train: %s" % (sys.exc_info()[0]))
        touch(failure_file)
        raise


def check_termination():
    if terminated:
        print("Exiting due to termination request")
        sys.exit(0)


def wait_till_fifo_exists(fname):
    print("Wait till FIFO available: %s" % (fname))
    while not os.path.exists(fname) and not terminated:
        time.sleep(0.1)
    check_termination()


def touch(fname):
    open(fname, "wa").close()


def on_terminate(signum, frame):
    print("caught termination signal, exiting gracefully...")
    global terminated
    terminated = True


def trap_signal():
    signal.signal(signal.SIGTERM, on_terminate)
    signal.signal(signal.SIGINT, on_terminate)


if __name__ == "__main__":
    # As per the SageMaker container spec, the algo takes a 'train' parameter.
    # We will simply ignore this in this dummy implementation.
    main()
