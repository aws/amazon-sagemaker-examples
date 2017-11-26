import os
import sys
import tarfile
from six.moves import urllib
from ipywidgets import FloatProgress
from IPython.display import display

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def cifar10_download(data_dir='/tmp/cifar10_data', print_progress=True):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.exists(os.path.join(data_dir, 'cifar-10-batches-bin')):
        print('cifar dataset already downloaded')
        return

    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        f = FloatProgress(min=0, max=100)
        display(f)
        sys.stdout.write('\r>> Downloading %s ' % (filename))        

        def _progress(count, block_size, total_size):
            if print_progress:
                f.value = 100.0 * count * block_size / total_size

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(data_dir)
