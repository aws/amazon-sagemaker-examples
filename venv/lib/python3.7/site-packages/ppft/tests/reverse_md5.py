#!/usr/bin/env python
# File: reverse_md5.py
# Author: Vitalii Vanovschi
# Desc: This program demonstrates parallel computations with pp module
# It tries to reverse an md5 hash in parallel
# Parallel Python Software: http://www.parallelpython.com

import math
import sys
import hashlib
import ppft


def md5test(hash, start, end):
    """Calculates md5 of the integers between 'start' and 'end' and
       compares it with 'hash'"""
    from hashlib import md5
    from ppft.common import b_
    for x in range(start, end):
        if md5(b_(str(x))).hexdigest() == hash:
            return x


print("""Usage: python reverse_md5.py [ncpus]
    [ncpus] - the number of workers to run in parallel,
    if omitted it will be set to the number of processors in the system
""")

# tuple of all parallel python servers to connect with
#ppservers = ("*",) # auto-discover
#ppservers = ("10.0.0.1","10.0.0.2") # list of static IPs
ppservers = ()

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = ppft.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = ppft.Server(ppservers=ppservers)

print("Starting ppft with %s workers" % job_server.get_ncpus())

#Calculates md5 hash from the given number
import hashlib
from ppft.common import b_
hash = hashlib.md5(b_("1829182")).hexdigest()
print("hash = %s" % hash)
#Now we will try to find the number with this hash value

start = 1
end = 2000000

# Since jobs are not equal in the execution time, division of the problem
# into a 128 of small subproblems leads to a better load balancing
parts = 128

step = int((end - start) / parts + 1)
jobs = []

for index in range(parts):
    starti = start+index*step
    endi = min(start+(index+1)*step, end)
    # Submit a job which will test if a number in the range starti-endi
    # has given md5 hash
    # md5test - the function
    # (hash, starti, endi) - tuple with arguments for md5test
    # () - tuple with functions on which function md5test depends
    # () - tuple with module names to be imported before md5test execution
    # jobs.append(job_server.submit(md5test, (hash, starti, endi),
    # globals=globals()))
    jobs.append(job_server.submit(md5test, (hash, starti, endi), (), ()))

# Retrieve results of all submited jobs
for job in jobs:
    result = job()
    if result:
        break

# Print the results
if result:
    print("Reverse md5 for %s is %s" % (hash, result))
else:
    print("Reverse md5 for %s has not been found" % hash)

job_server.print_stats()

# Properly finalize all tasks (not necessary)
job_server.wait()

# Parallel Python Software: http://www.parallelpython.com
