from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("Number of MPI processes that will talk to each other:", size)


def point_to_point():
    """Point to point communication
    Send a numpy array (buffer like object) from rank 0 to rank 1
    """
    if rank == 0:
        print("point to point")
        data = np.array([0, 1, 2], dtype=np.intc)  # int in C

        # remember the difference between
        # Upper case API and lower case API
        # Basically uppper case API directly calls C API
        # so it is fast
        # checkout https://mpi4py.readthedocs.io/en/stable/

        comm.Send([data, MPI.INT], dest=1)
    elif rank == 1:
        print(f"Hello I am rank {rank}")
        data = np.empty(3, dtype=np.intc)
        comm.Recv([data, MPI.INT], source=0)
        print("I received some data:", data)

    if rank == 0:
        time.sleep(1)  # give some buffer time for execution to complete
        print("=" * 50)
    return


def broadcast():
    """Broadcast a numpy array from rank 0 to others"""

    if rank == 0:
        print(f"Broadcasting from rank {rank}")
        data = np.arange(10, dtype=np.intc)
    else:
        data = np.empty(10, dtype=np.intc)

    comm.Bcast([data, MPI.INT], root=0)
    print(f"Data at rank {rank}", data)

    if rank == 0:
        time.sleep(1)
        print("=" * 50)
    return


def gather_reduce_broadcast():
    """Gather numpy arrays from all ranks to rank 0
    then take average and broadcast result to other ranks

    It is a useful operation in distributed training:
    train a model in a few MPI workers with different
    input data, then take average weights on rank 0 and
    synchroinze weights on other ranks
    """

    # stuff to gather at each rank
    sendbuf = np.zeros(10, dtype=np.intc) + rank
    recvbuf = None

    if rank == 0:
        print("Gather and reduce")
        recvbuf = np.empty([size, 10], dtype=np.intc)
    comm.Gather(sendbuf, recvbuf, root=0)

    if rank == 0:
        print(f"I am rank {rank}, data I gathered is: {recvbuf}")

        # take average
        # think of it as a prototype of
        # average weights, average gradients etc
        avg = np.mean(recvbuf, axis=0, dtype=np.float)

    else:
        # get averaged array from rank 0
        # think of it as a prototype of
        # synchronizing weights across different MPI procs
        avg = np.empty(10, dtype=np.float)

    # Note that the data type is float here
    # because we took average
    comm.Bcast([avg, MPI.FLOAT], root=0)

    print(f"I am rank {rank}, my avg is: {avg}")
    return


if __name__ == "__main__":
    point_to_point()
    broadcast()
    gather_reduce_broadcast()
