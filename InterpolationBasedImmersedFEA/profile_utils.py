"""
The ``profile_utils`` module 
---------------------
contains profiling tools for parallel tests. 
"""
import cProfile, pstats, io
from mpi4py import MPI as pyMPI

def profile_separate(filename=None, comm=pyMPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.Get_rank())
                pr.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator
