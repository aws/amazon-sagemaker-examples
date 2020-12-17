# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this
# file except in compliance with the License. A copy of the License is located at
#
#    http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from __future__ import print_function
import multiprocessing as mp
import numpy as np
import gzip as gz
import socket
import struct
from functools import partial

import datetime

# Script Parameters
NUM_PROCESSES = 4
ASN_FILE = "ip2asn-v4-u32.tsv.gz"

WORKER_DONE_SIGNAL = "DONE"
MAX_QUEUE_SIZE = 10000

# p_travel hyperparamters
P_TRAVEL_ALPHA = 1
P_TRAVEL_BETA = 30

# p_home hyperparamters
P_HOME_ALPHA = 3
P_HOME_BETA = 5

# Num events per user
NUM_EVENTS_PARETO_A = 1
NUM_EVENTS_PARETO_LOC = 1
NUM_EVENTS_PARETO_SCALE = 50
MAX_NUM_OF_EVENTS = 10000

# Home work proximity
HOME_WORK_DISTANCE_UNIFORM_WINDOW = 50000

# Not just pick IP from home/work network, but perhaps few others around them
HOME_WORK_LOCALITY_WINDOW_SIGMA = 10

# Log Formats
LOG_DATE_FORMAT = '%d/%b/%Y:%H:%M:%S +0000'
LOG_FORMAT = '{} - {} [{}] "GET /login_success HTTP/1.1" 200 476 "-" \
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/555.33 \
(KHTML, like Gecko) Chrome/1.1.1111.100 Safari/555.355"\n'
END_DATE = datetime.datetime(2018, 11, 14)


def ip2int(ip):
    """Convert an IP string to long"""
    packed_ip = socket.inet_aton(ip)
    return struct.unpack("!I", packed_ip)[0]


def load_asn_list(file_name=ASN_FILE):
    """Load and return Autonomous Systems and corresponding IP Ranges."""
    asn_list = []
    with gz.open(file_name, 'r') as f:
        for ln in f:
            tks = ln.strip().split()
            cur_asn = {
                "begin": int(tks[0]),
                "end": int(tks[1]),
                "asn": tks[2],
                "country": tks[3]
            }
            asn_list.append(cur_asn)

    return asn_list


# Load ASN list into memory
global asn_list
asn_list = load_asn_list()
print("Loaded ASN List: {} ASNs.".format(len(asn_list)))


def int2ip(n):
    """Convert an long to IP string."""
    packet_ip = struct.pack("!I", n)
    return socket.inet_ntoa(packet_ip)


def draw_ip_from_asn(asn):
    """Draw an IP address from given ASN uniform at random."""
    ip_address_int = np.random.randint(low=asn["begin"], high=asn["end"]+1)
    return int2ip(ip_address_int)


def draw_ip():
    """Draw a random IP address from random ASN all uniform at random."""
    asn = np.random.randint(len(asn_list))
    return draw_ip_from_asn(asn_list[asn])


def draw_user_ip(home_asn, work_asn, p_travel, p_home):
    """Draw IP address from user's distributed defined by input parameters.

    When drawing an IP address for a login event, we first draw whether a user
        is 'traveling' or not. If they are traveling, we assign them a
        random ASN to connect to.

    If they are not traveling, we then draw whether or not the user is at home
        or at work that day. We assume the user travels within a viscinity of
        their home/work. Thus, we draw an IP address from within a radius around
        their home/work radius.

    Once we have the ASN a user is using for this access, we uniformly sample
        from the ASN's IP range for the assigned IP address.

    :param home_asn: (int) the ASN idx corresponding to the user's home.
    :param work_asn: (int) the ASN idx corresponding to the user's work.
    :param p_travel: (float) the probability that the user is traveling.
    :param p_home: (float) the probability that the user is at home versus work.
    :return (string) an IPv4 address in dot-notation.
    """
    # If user is traveling, pick a random ASN
    if np.random.rand() < p_travel:
        cur_asn = np.random.randint(len(asn_list))
    else:
        # User is at home or at work
        home_or_work = home_asn if np.random.rand() < p_home else work_asn

        # Assume user travels locally around work or home
        cur_asn = np.random.normal(
            loc=home_or_work,
            scale=HOME_WORK_LOCALITY_WINDOW_SIGMA)
        cur_asn = int(cur_asn) % len(asn_list)

    cur_ip = draw_ip_from_asn(asn_list[cur_asn])
    return cur_ip


def generate_user_asns(num_asn, home_work_distance_uniform_window=HOME_WORK_DISTANCE_UNIFORM_WINDOW):
    """Generate home and work ASN ids for a user.

    We assume each user to be associated with two different ASNs:
        - one associated with their 'home' network and
        - one associated with their 'work' network.

    :param num_asn: (int) the number of ASNs to draw from
    :param home_work_distance_uniform_window: (int) the max distance between
        an individual's home and work
    :return (tuple[int]) the user's home and work ASN idx
    """
    home = np.random.randint(num_asn)

    work = home
    while work == home:
        work_low = home - home_work_distance_uniform_window
        work_high = home + home_work_distance_uniform_window
        work = np.random.randint(low=work_low, high=work_high)
        work = work % num_asn

    return home, work


def generate_user_login_events(queue, user_id, max_num_of_events):
    """Genrate login events for a user according to generative model.

    We assume each user has an ASN associated with their home and their work.
        Users has an assigned probability of being at either work or at home.

    Each user also has an associated probability of their likelihood to be
        'traveling', which means they are neither at home nor work.

    Based on these probabilities, draw an IP address uniformly
    from their currently used ASN. Then append the simulated user activity
    list((user, ip address)) to the queue.
    """
    # Initialize random seed for user
    np.random.seed(seed=np.random.randint(100000) + user_id)

    # Select an ASN for the user's home and work network
    home, work = generate_user_asns(len(asn_list))

    # Sample how active the user is based on a Pareto distribution
    num_events = int(
        NUM_EVENTS_PARETO_SCALE *
        (np.random.pareto(NUM_EVENTS_PARETO_A) + NUM_EVENTS_PARETO_LOC)
    )
    num_events = min(max_num_of_events, num_events)

    # Sample traveling probability for this user
    p_travel = np.random.beta(P_TRAVEL_ALPHA, P_TRAVEL_BETA)

    # Sample staying home probability for this user.
    # If not travelling, user is either at home or work
    p_home = np.random.beta(P_HOME_ALPHA, P_HOME_BETA)

    all_events = []
    for i in range(num_events):
        cur_ip = draw_user_ip(home, work, p_travel, p_home)
        event_log = (user_id, cur_ip)
        all_events.append(event_log)

    queue.put(all_events)
    queue.put(WORKER_DONE_SIGNAL)


def generate_random_timestamp(end=END_DATE, days=10):
    """Generate a random datetime between num_days before start."""
    date_window_sec = int(datetime.timedelta(days=days).total_seconds())
    random_delta = np.random.randint(0, date_window_sec)
    return end - datetime.timedelta(seconds=random_delta)


def format_event_as_log(event):
    timestamp = generate_random_timestamp()
    timestamp_str = timestamp.strftime(LOG_DATE_FORMAT)

    user_id, ip_address = event
    user_name = 'user_{}'.format(user_id)
    log_line = LOG_FORMAT.format(ip_address, user_name, timestamp_str)
    return log_line


def queue_to_file(queue, file_name, num_users):
    if file_name.startswith('s3://'):
        import s3fs
        s3filesystem = s3fs.S3FileSystem()
        fp = s3filesystem.open(file_name, 'w')
    else:
        fp = open(file_name, 'w')

    try:
        from tqdm import tqdm
        pbar = tqdm(total=num_users, unit='users')
    except ImportError:
        pbar = None

    num_finished_workers = 0
    while num_finished_workers < num_users:
        event = queue.get()

        if event == WORKER_DONE_SIGNAL:
            num_finished_workers += 1
            if pbar:
                pbar.update(1)
        else:
            for e in event:
                fp.write(format_event_as_log(e))

    fp.close()


def generate_dataset(num_users, file_name, max_num_of_events=MAX_NUM_OF_EVENTS,
                     chunk_size=1000, num_processes=NUM_PROCESSES):
    """Generate user traffic for specified number of users.

    We generate simulated traffic for users under the following scenario:

    - When a user connects to the internet, they typically receive an
        IP addresses from an entity such as an Internet service provider (ISP).
        Each ISP has an officially registered autonomous system number (ASN),
        which corresponds to a set of IP addresses they can allocate. Depending
        on where a user is connecting from, they connect to a different ASN.
    - We define each user to have a static 'home' and 'work' location with
        corresponding ASNs.
    - Each user has a probability of traveling. If they are not traveling,
        the user has a static probability of being at either home or work.
    - If a user is traveling, we sample a login even from a random ASN.

    Based on if the user is traveling, at home, or at work we simulate the
    user's current ASN. Then, using an exemplary ASN to IP range look-up table,
    we uniformly sample an IP address from their assigned ASN. We repeate this
    for a variable amount of time to generate a user's login history.
    """

    # Create a pool of processes that will:
    # generate user activity and push to a common queue
    pool = mp.Pool(num_processes)
    manager = mp.Manager()
    queue = manager.Queue(MAX_QUEUE_SIZE)

    # Kick off process that will save queued events to a file
    print("Starting User Activity Simulation")
    p = mp.Process(target=queue_to_file, args=(queue, file_name, num_users,))
    p.start()

    # Break up tasks by having a pool process a chunk of users at a time
    chunk_size = min(chunk_size, num_users)
    for i in range(num_users // chunk_size):
        chunk_start = i * chunk_size
        user_ids_chunk = range(chunk_start, chunk_start+chunk_size)
        generate_func = partial(
            generate_user_login_events,
            queue,
            max_num_of_events=max_num_of_events)
        pool.map(generate_func, user_ids_chunk)

    p.join()
    pool.close()
    pool.join()
    print("Finished simulating web activity for {} users.".format(num_users))
