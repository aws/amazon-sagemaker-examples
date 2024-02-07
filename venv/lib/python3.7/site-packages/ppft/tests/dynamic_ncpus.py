#!/usr/bin/env python
# File: dynamic_ncpus.py
# Author: Vitalii Vanovschi
# Desc: This program demonstrates parallel computations with pp module
# and dynamic cpu allocation feature.
# Program calculates the partial sum 1-1/2+1/3-1/4+1/5-1/6+...
# (in the limit it is ln(2))
# Parallel Python Software: http://www.parallelpython.com

import math
import sys
import time
import ppft as pp


def part_sum(start, end):
    """Calculates partial sum"""
    sum = 0
    for x in range(start, end):
        if x % 2 == 0:
            sum -= 1.0 / x
        else:
            sum += 1.0 / x
    return sum

print("""Usage: python dynamic_ncpus.py""")
print("")

start = 1
end = 20000000

# Divide the task into 64 subtasks
parts = 64
step = int((end - start) / parts + 1)

# Create jobserver
job_server = pp.Server()

# Execute the same task with different amount of active workers
# and measure the time
for ncpus in (1, 2, 4, 8, 16, 1):
    job_server.set_ncpus(ncpus)
    jobs = []
    start_time = time.time()
    print("Starting pp with %s workers" % job_server.get_ncpus())
    for index in range(parts):
        starti = start+index*step
        endi = min(start+(index+1)*step, end)
        # Submit a job which will calculate partial sum
        # part_sum - the function
        # (starti, endi) - tuple with arguments for part_sum
        # () - tuple with functions on which function part_sum depends
        # () - tuple with module names which must be
        #      imported before part_sum execution
        jobs.append(job_server.submit(part_sum, (starti, endi)))

    # Retrieve all the results and calculate their sum
    part_sum1 = sum([job() for job in jobs])
    # Print the partial sum
    print("Partial sum is %s | diff = %s" % (part_sum1, math.log(2) - part_sum1))

    print("Time elapsed:  %s s" % (time.time() - start_time))
    print("")
job_server.print_stats()

# Parallel Python Software: http://www.parallelpython.com
