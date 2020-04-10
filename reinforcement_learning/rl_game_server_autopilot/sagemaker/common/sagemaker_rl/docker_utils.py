import socket
import time

def get_ip_from_host(timeout=100, host_name=None):
    counter = 0
    ip_address = None

    if not host_name:
        host_name = socket.gethostname()
        print("Fetching IP for hostname: %s" % host_name)
    while counter < timeout and not ip_address:
        try:
            ip_address = socket.gethostbyname(host_name)
            break
        except Exception as e:
            counter += 1
            time.sleep(1)

    if counter == timeout and not ip_address:
        error_string = "Platform Error: Could not retrieve IP address \
        for %s in past %s seconds" % (host_name, timeout)
        raise RuntimeError(error_string)

    return ip_address