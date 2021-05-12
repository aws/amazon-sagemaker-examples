import subprocess

import boto3

SAGEMAKER_DOCKER_MARKOV_PATH = "/opt/amazon/markov"
MARKOV_FOLDER = "./src/markov"


def run_cmd(cmd_args, change_working_directory="./", shell=False, executable=None):
    """
    Function used to execute the shell commands

    return (returncode, result)
    : returncode: int - This contains the 0/1 based on whether command
    : result: list - This is the output result by executing the command as a list

    :param cmd_args: list - This is the list of commands to be appended in case
        of shell=True, else executed as pipes.
    :param change_working_directory: string - This is to execute the command in different
        directory
    :param shell: bool - This is to say if the command to be executed as shell or not
    """
    cmd = " ".join(map(str, cmd_args))
    print(cmd)
    process = subprocess.Popen(
        cmd if shell else cmd_args,
        cwd=change_working_directory,
        shell=shell,
        executable=executable,
        stdout=subprocess.PIPE,
    )
    result = list()
    for line in iter(process.stdout.readline, b""):
        result.append(line.decode("utf-8").rstrip())
    process.communicate()
    return process.returncode, result


def get_sagemaker_docker(repository_short_name):
    """
    If the sagemaker docker is already created it picks the most recently created docker
    with the Repository name you created earlier (Sagemaker docker). If not present,
    creates one sagemaker docker and returns its docker id.

    return (docker_id)
    :docker_id: string - This is the sagemaker docker id.
    """
    _, docker_ids = run_cmd(
        [r"docker images {} | sed -n 2,2p".format(repository_short_name)], shell=True
    )
    if docker_ids and docker_ids[0]:
        docker_id = [docker for docker in docker_ids[0].split(" ") if docker != ""]
        print("Sagemaker docker id : {}".format(docker_id[2]))
        return docker_id[2]
    raise Exception("SageMaker docker not found. Please check.")


def copy_to_sagemaker_container(sagemaker_docker_id, repository_short_name):
    """
    This function will copy the contents to the sagemaker container. This is required because,
    the docker would alread be created with the original code and we need a docker container
    to copy the files from the src package to the docker container.
    """
    _, docker_containers = run_cmd(["docker run -d -t {}".format(sagemaker_docker_id)], shell=True)
    #
    # Docker cp does not overwrite the files if modified. This is fixed in the
    # newer version. But the current version does not. Hence deleting the folder
    # and then copying the files to the container
    #

    # Copy Markov package
    # Deleting markov folder in the sagemaker container
    run_cmd(
        [
            "docker exec -d {0} rm -rf {1}".format(
                docker_containers[0], SAGEMAKER_DOCKER_MARKOV_PATH
            )
        ],
        shell=True,
    )
    # Copying markov folder to the sagemaker container
    run_cmd(
        [
            "docker cp {0} {1}:{2}".format(
                MARKOV_FOLDER, docker_containers[0], SAGEMAKER_DOCKER_MARKOV_PATH
            )
        ],
        shell=True,
    )
    print("============ Copied Markov scripts to sagemaker docker ============ \n ")

    docker_processes = run_cmd(["docker ps -l|sed -n 2,2p"], shell=True)
    docker_ps = [
        docker_process
        for docker_process in docker_processes[1][0].split(" ")
        if docker_process != ""
    ][0]

    # Committing all the changes to the docker
    run_cmd([r"docker commit {0} {1}".format(docker_ps, repository_short_name)], shell=True)
    print("============ Commited all the changes to docker ============ \n ")


def get_custom_image_name(custom_image_name):
    session = boto3.Session()
    aws_account = session.client("sts").get_caller_identity()["Account"]
    aws_region = session.region_name
    ecr_repo = "%s.dkr.ecr.%s.amazonaws.com" % (aws_account, aws_region)
    ecr_tag = "%s/%s" % (ecr_repo, custom_image_name)
    return ecr_tag
