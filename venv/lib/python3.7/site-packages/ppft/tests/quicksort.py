#!/usr/bin/env python
# File: sum_primes.py
# Author: Vitalii Vanovschi
# Desc: This program demonstrates parallel version of quicksort algorithm
# implemented using pp module
# Parallel Python Software: http://www.parallelpython.com

import sys, random
import ppft as pp

def quicksort(a, n=-1, srv=None):
    if len(a) <= 1:
        return a
    if n:
        return quicksort([x for x in a if x < a[0]], n-1, srv) \
                + [a[0]] + quicksort([x for x in a[1:] if x >= a[0]], n-1, srv)
    else:
        return [srv.submit(quicksort, (a,))]
    

print("""Usage: python quicksort.py [ncpus]
    [ncpus] - the number of workers to run in parallel, 
    if omitted it will be set to the number of processors in the system
""")

# tuple of all parallel python servers to connect with
#ppservers = ("*",)
#ppservers = ("10.0.0.1",)
ppservers = ()

if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print("Starting pp with %s workers" % job_server.get_ncpus())

n = 1000000
input = []
for i in range(n):
    input.append(random.randint(0,100000))

# set n to a positive integer to create 2^n PP jobs 
# or to -1 to avoid using PP

# 32 PP jobs
n = 5

# no PP
#n = -1

outputraw = quicksort(input, n, job_server)

output = []
for x in outputraw:
    if callable(x):
        output.extend(x())
    else:
        output.append(x)

print("first 30 numbers in increasing order: %s" % output[:30])

job_server.print_stats()

# Parallel Python Software: http://www.parallelpython.com
