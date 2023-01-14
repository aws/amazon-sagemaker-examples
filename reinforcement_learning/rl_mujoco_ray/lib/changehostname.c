#include <stdio.h>
#include <string.h>

/*
 * Modifies gethostname to return algo-1, algo-2, etc. when running on SageMaker.
 *
 * Without this gethostname() on SageMaker returns 'aws', leading NCCL/MPI to think there is only one host,
 * not realizing that it needs to use NET/Socket.
 *
 * When docker container starts we read 'current_host' value  from /opt/ml/input/config/resourceconfig.json
 * and replace PLACEHOLDER_HOSTNAME with it before compiling this code into a shared library.
 */
int gethostname(char *name, size_t len)
{
  const char *val = PLACEHOLDER_HOSTNAME;
  strncpy(name, val, len);
  return 0;
}
