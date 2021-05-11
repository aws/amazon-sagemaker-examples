#!/usr/bin/env python

"""
Script to do faster development of the SimulationApplication.

1. Untar the output.tar.gz that is available after building the SimulationApplication.
2. Creates the tar that is compitable with the colcon build

# To untar the SimApp .tar.gz
python3 sim_app_bundler.py --untar ./deepracer-sim-app.tar.gz

# After modification to create .tar.gz compatable with colcon build (For RoboMaker)
python3 sim_app_bundler.py --tar

# Clean the build
python3 sim_app_bundler.py --clean
"""
import argparse
import os
import shutil
import subprocess
import tarfile

UNTARRED_SIM_APP_OUTPUT_PATH = "build/simapp/"
BUILD_PATH = "build"
if not os.path.exists(BUILD_PATH):
    os.mkdir(BUILD_PATH)

"""
#
# AwsSilverstoneSimulationApplication
# This package mainly contains the RoboMaker launcher files
# Package path: https://code.amazon.com/packages/AwsSilverstoneSimulationApplication/trees/mainline
# 

|- bundle/opt/install/deepracer_simulation_environment/share/deepracer_simulation_environment/
|- bundle/opt/install/deepracer_simulation_environment/lib/deepracer_simulation_environment/

"""

"""
#
# AwsSilverstoneMarkovScripts
# This package mainly contains the sagemaker related files
# Package path: https://code.amazon.com/packages/AwsSilverstoneMarkovScripts/trees/mainline

|- bundle/opt/install/sagemaker_rl_agent/lib/python3.5/site-packages/

"""

"""
#
# AwsSilverstoneSimulationPhysics related constants
# This package deals with the Physics of the car. Spunging the car, chassis details, mass etc.
# Package path: https://code.amazon.com/packages/AwsSilverstoneSimulationPhysics/trees/mainline

|- bundle/opt/install/deepracer_simulation_environment/share/deepracer_simulation_environment/
    meshes & urdf
"""

"""
#
# AwsSilverstoneSimulationTracks
# This package deals with the Simulation environment. The car tracks, lighting etc.
# Package path: https://code.amazon.com/packages/AwsSilverstoneSimulationTracks/trees/mainline

|- bundle/opt/install/deepracer_simulation_environment/share/deepracer_simulation_environment/
    meshes, models, routes, worlds
"""

# Simulation App file structure
SIM_APP_OUTPUT_FILE_NAME = "output.tar.gz"
BUNDLE_PATH = os.path.join(UNTARRED_SIM_APP_OUTPUT_PATH, "bundle")
BUNDLE_TAR_PATH = os.path.join(UNTARRED_SIM_APP_OUTPUT_PATH, "bundle.tar")
METADATA_TAR_PATH = os.path.join(UNTARRED_SIM_APP_OUTPUT_PATH, "metadata.tar")


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


def _is_path_exists(path):
    """
    Private function to check if the path exists

    return (is_path_exists)
    : is_path_exists: bool - True if exists else False

    :param path: string - Operating system path
    """
    if os.path.exists(path):
        return True
    return False


def untar_simapp_output(sim_app_output_path):
    """
    Function used to untar the AwsSilverstoneSimulation/build/output.tar.gz
    and put to the build location of the AwsSilverstoneSimulationTest/build/output.tar.gz
    """
    if not _is_path_exists(sim_app_output_path):
        raise Exception("Not able to find SimApp tar.gz file")

    # Untar output.tar.gz
    if not os.path.exists(UNTARRED_SIM_APP_OUTPUT_PATH):
        os.mkdir(UNTARRED_SIM_APP_OUTPUT_PATH)
    cmd = ["tar", "-xvf", sim_app_output_path, "-C", UNTARRED_SIM_APP_OUTPUT_PATH]
    run_cmd(cmd)

    # Untar bundle.tar
    os.mkdir(BUNDLE_PATH)
    cmd = ["tar", "-xvf", BUNDLE_TAR_PATH, "-C", BUNDLE_PATH]
    run_cmd(cmd)

    # Untar metadata.tar
    cmd = ["tar", "-xvf", METADATA_TAR_PATH, "-C", UNTARRED_SIM_APP_OUTPUT_PATH]
    run_cmd(cmd)


def clean_tar_files():
    """
    The intermediate .tar files created are removed
    """
    if os.path.exists(BUNDLE_TAR_PATH):
        os.remove(BUNDLE_TAR_PATH)
    if os.path.exists(METADATA_TAR_PATH):
        os.remove(METADATA_TAR_PATH)


def generate_archive_v1():
    """
    This function is copied and modified from the colcon build github
    Modified little to serve our purpose of untarring and re-tarring
    the output.tar.gz.

    Any changes to improve the performance of this function would lead
    to the failure to start the Robomaker job. Robomaker uses the
    colcon build to build there application and I am using the same
    function that creates the bundle .tar.gz

    Generate bundle archive.
    output.tar.gz
    |- version
    |- metadata.tar
    |- bundle.tar
    :param path_context: PathContext object including path configurations
    """
    print("Bundling the SimApp output.tar.gz")

    bundle_tar_path = os.path.join(BUILD_PATH, "bundle.tar")
    metadata_tar_path = os.path.join(BUILD_PATH, "metadata.tar")
    archive_tar_gz_path = os.path.join(BUILD_PATH, "output.tar.gz")

    if os.path.exists(bundle_tar_path):
        print("Removing previously build {}".format(bundle_tar_path))
        os.remove(bundle_tar_path)

    if os.path.exists(archive_tar_gz_path):
        print("Removing previously build {}".format(archive_tar_gz_path))
        os.remove(archive_tar_gz_path)

    with tarfile.open(metadata_tar_path, "w") as archive:
        archive.add(
            os.path.join(UNTARRED_SIM_APP_OUTPUT_PATH, "installers.json"), arcname="installers.json"
        )

    _recursive_tar_in_path(bundle_tar_path, os.path.join(UNTARRED_SIM_APP_OUTPUT_PATH, "bundle"))

    with tarfile.open(archive_tar_gz_path, "w:gz", compresslevel=5) as archive:
        archive.add(os.path.join(UNTARRED_SIM_APP_OUTPUT_PATH, "version"), arcname="version")
        archive.add(metadata_tar_path, arcname=os.path.basename(metadata_tar_path))
        archive.add(bundle_tar_path, arcname=os.path.basename(bundle_tar_path))

    print("Removing the previously created tar files")
    os.remove(metadata_tar_path)
    os.remove(bundle_tar_path)

    print("============ Archiving completed. Available at build/Output.tar.gz============ \n")


def _recursive_tar_in_path(tar_path, path, *, mode="w"):
    """
    As the above generate_archive_v1, this is also copied from the
    colcon build github.

    Tar all files inside a directory.
    This function includes all sub-folders of path in the root of the tarfile
    :param tar_path: The output path
    :param path: path to recursively collect all files and include in
    tar
    :param mode: mode flags passed to tarfile
    """
    with tarfile.open(tar_path, mode) as tar:
        print("Creating tar of {path}".format(path=path))
        for name in os.listdir(path):
            some_path = os.path.join(path, name)
            tar.add(some_path, arcname=os.path.basename(some_path))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-u", "--untar", help="Untar the SimApp", type=str, required=False)
    parser.add_argument("-t", "--tar", action="store_true", help="Tar the simulation application")

    parser.add_argument(
        "-c", "--clean", action="store_true", help="Clean the untarred simulation application"
    )

    args, unknown = parser.parse_known_args()

    if args.untar:
        if os.path.exists(UNTARRED_SIM_APP_OUTPUT_PATH):
            print("Untarred version already exists. Please use --clean to clean the build")
        else:
            print("Untarring all files in ./{}".format(UNTARRED_SIM_APP_OUTPUT_PATH))
            os.mkdir(UNTARRED_SIM_APP_OUTPUT_PATH)
            untar_simapp_output(args.untar)
            clean_tar_files()

    if args.tar:
        generate_archive_v1()

    if args.clean:
        print("Deleting all files in ./{}".format(BUILD_PATH))
        shutil.rmtree(BUILD_PATH, ignore_errors=True)


if __name__ == "__main__":
    main()
