#!/usr/bin/env python
# File: callback.py
# Author: Vitalii Vanovschi
# Desc: This program demonstrates parallel computations with pp module
# using callbacks (available since pp 1.3).
# Program calculates the partial sum 1-1/2+1/3-1/4+1/5-1/6+...
# (in the limit it is ln(2))
# Parallel Python Software: http://www.parallelpython.com

import math
import time
import _thread as thread
import sys
import ppft as pp


class Sum(object):
    """Class for callbacks
    """

    def __init__(self):
        self.value = 0.0
        self.lock = thread.allocate_lock()

    def add(self, value):
        """
        the callback function
        """
        # we must use lock here because += is not atomic
        self.lock.acquire()
        self.value += value
        self.lock.release()


def part_sum(start, end):
    """Calculates partial sum"""
    sum = 0
    for x in range(start, end):
        if x % 2 == 0:
            sum -= 1.0 / x
        else:
            sum += 1.0 / x
    return sum


print("""Usage: python callback.py [ncpus]
    [ncpus] - the number of workers to run in parallel,
    if omitted it will be set to the number of processors in the system
""")

start = 1
end = 20000000

# Divide the task into 128 subtasks
parts = 128
step = int((end - start) / parts + 1)

# tuple of all parallel python servers to connect with
#ppservers = ("*",) # auto-discover
#ppservers = ("10.0.0.1","10.0.0.2") # list of static IPs
ppservers = ()

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print("Starting pp with %s workers" % job_server.get_ncpus())

# Create anm instance of callback class
sum = Sum()

# Execute the same task with different number
# of active workers and measure the time

start_time = time.time()
for index in range(parts):
    starti = start+index*step
    endi = min(start+(index+1)*step, end)
    # Submit a job which will calculate partial sum
    # part_sum - the function
    # (starti, endi) - tuple with arguments for part_sum
    # callback=sum.add - callback function
    job_server.submit(part_sum, (starti, endi), callback=sum.add)

#wait for jobs in all groups to finish
job_server.wait()

# Print the partial sum
print("Partial sum is %s | diff = %s" % (sum.value, math.log(2) - sum.value))

job_server.print_stats()

# Parallel Python Software: http://www.parallelpython.com
