#include <stdio.h>
#include <string.h>

/*
 * Modifies gethostname to return algo-1, algo-2, etc. when running on SageMaker.
 *
 * ChainerMN calls MPI's Get_processor_name() when initializing ranks.
 * OpenMPI's Get_processor_name() in turn calls gethostname().
 * https://github.com/open-mpi/ompi/blob/master/ompi/mpi/c/get_processor_name.c#L69
 * Without this, gethostname() on SageMaker returns 'aws', leading OpenMPI to think there is only one processor,
 * screwing up rank initialization and who knows what else (OpenMPI calls gethostname() liberally).
 *
 * The framework training code calls change-hostname.sh, passing in the 'real' hostname ('algo-1', 'algo-2', etc) to
 * replace "PLACEHOLDER_HOSTNAME" before compiling this code into a shared library.
 */
int gethostname(char *name, size_t len)
{
  const char *val = PLACEHOLDER_HOSTNAME;
  strncpy(name, val, len);
  return 0;
}
