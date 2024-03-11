#!/usr/bin/env python
# File: sum_primes_template.py
# Author: Vitalii Vanovschi
# Desc: This program demonstrates using pp template class
# It calculates the sum of prime numbers below a given integer in parallel
# Parallel Python Software: http://www.parallelpython.com

import math, sys
import ppft as pp


def isprime(n):
    """Returns True if n is prime and False otherwise"""
    if not isinstance(n, int):
        raise TypeError("argument passed to is_prime is not of 'int' type")
    if n < 2:
        return False
    if n == 2:
        return True
    max = int(math.ceil(math.sqrt(n)))
    i = 2
    while i <= max:
        if n % i == 0:
            return False
        i += 1
    return True


def sum_primes(n):
    """Calculates sum of all primes below given integer n"""
    return sum([x for x in range(2,n) if isprime(x)])


print("""Usage: python sum_primes.py [ncpus]
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
    job_server = pp.Server(ncpus, ppservers=ppservers)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=ppservers)

print("Starting pp with %s workers" % job_server.get_ncpus())

# Creates a template
# Template is created using all the parameters of the jobs except 
# the arguments of the function 
# sum_primes - the function
# (isprime,) - tuple with functions on which function sum_primes depends
# ("math",) - tuple with module names which must be imported 
#             before sum_primes execution
fn = pp.Template(job_server, sum_primes, (isprime,), ("math",))

# Submit a job of calulating sum_primes(100) for execution using 
# previously created template
# Execution starts as soon as one of the workers will become available
job1 = fn.submit(100)

# Retrieves the result calculated by job1
# The value of job1() is the same as sum_primes(100)
# If the job has not been finished yet, 
# execution will wait here until result is available
result = job1()

print("Sum of primes below 100 is %s" % result)

# The following submits 8 jobs and then retrieves the results
inputs = (100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700)
jobs = [(input, fn.submit(input)) for input in inputs]

for input, job in jobs:
    print("Sum of primes below %s is %s" % (input, job()))

job_server.print_stats()

# Parallel Python Software: http://www.parallelpython.com
