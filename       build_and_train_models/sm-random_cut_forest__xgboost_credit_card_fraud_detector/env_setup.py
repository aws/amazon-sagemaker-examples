import argparse
import json
from pathlib import Path
import subprocess
import logging
import sys

CURRENT_FILE = Path(__file__).resolve()
CURRENT_FOLDER = CURRENT_FILE.parent
LOGBOOK_FILE = Path(CURRENT_FOLDER, 'env_setup_logbook.json')
ROOT_PATHS = {'NotebookInstance': Path('/home/ec2-user'), 'Studio': Path('/root')}
BIN_PATHS = {'NotebookInstance': Path('/usr/bin'), 'Studio': Path('/opt/conda/bin')}


# Common setup

def get_sagemaker_mode() -> str:
    stack_outputs_file = Path(CURRENT_FOLDER, 'stack_outputs.json')
    with open(stack_outputs_file) as f:
        outputs = json.load(f)
    sagemaker_mode = outputs['SagemakerMode']
    if sagemaker_mode not in set(['Studio', 'NotebookInstance']):
        raise ValueError('SagemakerMode should be Studio or NotebookInstance. Check stack_outputs.json.')
    return sagemaker_mode


def get_executable() -> str:
    return sys.executable


def get_hostname() -> str:
    hostname_file = Path('/etc/hostname')
    if hostname_file.is_file():
        with open(hostname_file, 'r') as f:
            contents = f.readlines()
        assert len(contents) == 1
        hostname = contents[0].strip()
    else:
        logging.warning(f'Could not find {hostname_file}. Setting hostname to None.')
        hostname = None
    return hostname


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Setup environment for solution.')
    parser.add_argument('--force', action='store_true',)
    parser.add_argument('--log-level', type=str, default='INFO')
    args = parser.parse_args()
    return args


def read_file(file: str) -> str:
    with open(file, 'r') as f:
        return f.read()


def bash(cmd: str) -> subprocess.CompletedProcess:
    try:
        if logging.root.level > logging.DEBUG:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        else:
            stdout = sys.stdout
            stderr = sys.stderr
        process = subprocess.run(
            "set -e" + '\n' + cmd,
            shell=True,
            check=True,
            universal_newlines=True,  # same as text=True but support py3.6 too
            stdout=stdout,
            stderr=stderr
        )
    except subprocess.CalledProcessError as e:
        if logging.root.level > logging.DEBUG:
            logging.error('\n' + e.stderr)
        raise e
    return process


def logging_setup(level: str) -> None:
    level = logging.getLevelName(level)
    logging.basicConfig(stream=sys.stdout, level=level)


def env_setup() -> None:
    args = parse_args()
    logging_setup(args.log_level)
    sagemaker_mode = get_sagemaker_mode()
    if sagemaker_mode == 'Studio':
        hostname = get_hostname()
        logging.debug(f'hostname: {hostname}')
        executable = get_executable()
        logging.debug(f'executable: {executable}')
        if args.force or not in_logbook(hostname, executable):
            env_setup_studio()
            logging.info('Successfully setup environment.')
            add_to_logbook(hostname, executable)
        else:
            logging.info('Skipping. Already setup environment.')
    if sagemaker_mode == 'NotebookInstance':
        if args.force:
            env_setup_notebook_instance()
            logging.info('Successfully setup environment.')
        else:
            logging.info('Skipping. Already setup environment.')


def in_logbook(hostname: str, executable: str) -> bool:
    if LOGBOOK_FILE.is_file():
        with open(LOGBOOK_FILE, 'r') as f:
            logbook = json.load(f)
        for entry in logbook:
            if (entry['hostname'] == hostname) and (entry['executable'] == executable):
                return True
        logging.debug('Could not find a matching entry in logbook.')
        return False
    else:
        logging.debug(f'Could not find logbook at {LOGBOOK_FILE}.')
        return False


def add_to_logbook(hostname: str, executable: str) -> None:
    if (hostname is None) or (executable is None):
        logging.warn('Could not add to logbook because either hostname or executable is empty.')
    else:
        new_entry = {'hostname': hostname, 'executable': executable}
        if LOGBOOK_FILE.is_file():
            with open(LOGBOOK_FILE, 'r') as f:
                logbook = json.load(f)
        else:
            logbook = []
        for entry in logbook:
            if (entry['hostname'] == hostname) and (entry['executable'] == executable):
                return  # don't need to add since already in logbook
        logbook.append(new_entry)
        with open(LOGBOOK_FILE, 'w') as f:
            json.dump(logbook, f)


# Solution specific setup

def env_setup_notebook_instance() -> None:
    logging.info('Starting environment setup for Notebook Instance.')
    py_exec = get_executable()

    logging.info('Uninstalling Python packages installed with distutils.')
    bash("""
    # fix to upgrade `docutils` that was installed with `distutils` (hence pip can't uninstall)
    rm -rf /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/docutils
    rm -rf /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages/docutils-*
    """)

    # This is due to papermill->black causing an inconsistency
    logging.info("Removing incompatible package")
    bash(f"""
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    {py_exec} -m pip uninstall -y enum34
    """)

    logging.info('Upgrading pip packages.')
    bash(f"""
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    {py_exec} -m pip  install --upgrade pyyaml --ignore-installed
    """)

    logging.info('Installing pip packages.')
    bash(f"""
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    {py_exec} -m pip install -r {CURRENT_FOLDER}/notebooks/requirements.txt
    {py_exec} -m pip install -e {CURRENT_FOLDER}/notebooks/
    """)


def env_setup_studio() -> None:
    logging.info('Starting environment setup for Studio.')
    py_exec = get_executable()

    logging.info('Installing local packages.')
    bash(f"""
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    {py_exec} -m pip install -e {CURRENT_FOLDER}/notebooks/
    """)

    logging.info('Completed environment setup for Studio.')


if __name__ == "__main__":
    env_setup()
