import os
import urllib.request
import logging

def download(url):
    filename = os.path.split(url)[1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    else:
        logging.warning(f"{filename} already exists. Skipping download.")