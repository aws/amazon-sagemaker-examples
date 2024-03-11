# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Placeholder docstring"""
from __future__ import absolute_import

import os
import subprocess
import tempfile
import warnings
import six
from six.moves import urllib


def git_clone_repo(git_config, entry_point, source_dir=None, dependencies=None):
    """Git clone repo containing the training code and serving code.

    This method also validate ``git_config``, and set ``entry_point``,
    ``source_dir`` and ``dependencies`` to the right file or directory in the
    repo cloned.

    Args:
        git_config (dict[str, str]): Git configurations used for cloning files,
            including ``repo``, ``branch``, ``commit``, ``2FA_enabled``,
            ``username``, ``password`` and ``token``. The ``repo`` field is
            required. All other fields are optional. ``repo`` specifies the Git
            repository where your training script is stored. If you don't
            provide ``branch``, the default value 'master' is used. If you don't
            provide ``commit``, the latest commit in the specified branch is
            used. ``2FA_enabled``, ``username``, ``password`` and ``token`` are
            for authentication purpose. If ``2FA_enabled`` is not provided, we
            consider 2FA as disabled.

            For GitHub and GitHub-like repos, when SSH URLs are provided, it
            doesn't matter whether 2FA is enabled or disabled; you should either
            have no passphrase for the SSH key pairs, or have the ssh-agent
            configured so that you will not be prompted for SSH passphrase when
            you do 'git clone' command with SSH URLs. When https URLs are
            provided: if 2FA is disabled, then either token or username+password
            will be used for authentication if provided (token prioritized); if
            2FA is enabled, only token will be used for authentication if
            provided. If required authentication info is not provided, python
            SDK will try to use local credentials storage to authenticate. If
            that fails either, an error message will be thrown.

            For CodeCommit repos, 2FA is not supported, so '2FA_enabled' should
            not be provided. There is no token in CodeCommit, so 'token' should
            not be provided too. When 'repo' is an SSH URL, the requirements are
            the same as GitHub-like repos. When 'repo' is an https URL,
            username+password will be used for authentication if they are
            provided; otherwise, python SDK will try to use either CodeCommit
            credential helper or local credential storage for authentication.
        entry_point (str): A relative location to the Python source file which
            should be executed as the entry point to training or model hosting
            in the Git repo.
        source_dir (str): A relative location to a directory with other training
            or model hosting source code dependencies aside from the entry point
            file in the Git repo (default: None). Structure within this
            directory are preserved when training on Amazon SageMaker.
        dependencies (list[str]): A list of relative locations to directories
            with any additional libraries that will be exported to the container
            in the Git repo (default: []).

    Returns:
        dict: A dict that contains the updated values of entry_point, source_dir
        and dependencies.

    Raises:
        CalledProcessError: If 1. failed to clone git repo
                               2. failed to checkout the required branch
                               3. failed to checkout the required commit
        ValueError: If 1. entry point specified does not exist in the repo
                       2. source dir specified does not exist in the repo
                       3. dependencies specified do not exist in the repo
                       4. wrong format is provided for git_config
    """
    if entry_point is None:
        raise ValueError("Please provide an entry point.")
    _validate_git_config(git_config)
    dest_dir = tempfile.mkdtemp()
    _generate_and_run_clone_command(git_config, dest_dir)

    _checkout_branch_and_commit(git_config, dest_dir)

    updated_paths = {
        "entry_point": entry_point,
        "source_dir": source_dir,
        "dependencies": dependencies,
    }

    # check if the cloned repo contains entry point, source directory and dependencies
    if source_dir:
        if not os.path.isdir(os.path.join(dest_dir, source_dir)):
            raise ValueError("Source directory does not exist in the repo.")
        if not os.path.isfile(os.path.join(dest_dir, source_dir, entry_point)):
            raise ValueError("Entry point does not exist in the repo.")
        updated_paths["source_dir"] = os.path.join(dest_dir, source_dir)
    else:
        if os.path.isfile(os.path.join(dest_dir, entry_point)):
            updated_paths["entry_point"] = os.path.join(dest_dir, entry_point)
        else:
            raise ValueError("Entry point does not exist in the repo.")
    if dependencies is not None:
        updated_paths["dependencies"] = []
        for path in dependencies:
            if os.path.exists(os.path.join(dest_dir, path)):
                updated_paths["dependencies"].append(os.path.join(dest_dir, path))
            else:
                raise ValueError("Dependency {} does not exist in the repo.".format(path))
    return updated_paths


def _validate_git_config(git_config):
    """Validates the git configuration.

    Checks all configuration values except 2FA_enabled are string types. The
    2FA_enabled configuration should be a boolean.

    Args:
        git_config: The configuration to validate.
    """
    if "repo" not in git_config:
        raise ValueError("Please provide a repo for git_config.")
    for key in git_config:
        if key == "2FA_enabled":
            if not isinstance(git_config["2FA_enabled"], bool):
                raise ValueError("Please enter a bool type for 2FA_enabled'.")
        elif not isinstance(git_config[key], six.string_types):
            raise ValueError("'{}' must be a string.".format(key))


def _generate_and_run_clone_command(git_config, dest_dir):
    """Check if a git_config param is valid.

    If it is valid, create the command to git, clone the repo, and run it.

    Args:
        git_config ((dict[str, str]): Git configurations used for cloning files,
            including ``repo``, ``branch`` and ``commit``.
        dest_dir (str): The local directory to clone the Git repo into.

    Raises:
        CalledProcessError: If failed to clone git repo.
    """
    if git_config["repo"].startswith("https://git-codecommit") or git_config["repo"].startswith(
        "ssh://git-codecommit"
    ):
        _clone_command_for_codecommit(git_config, dest_dir)
    else:
        _clone_command_for_github_like(git_config, dest_dir)


def _clone_command_for_github_like(git_config, dest_dir):
    """Check if a git_config param representing a GitHub (or like) repo is valid.

    If it is valid, create the command to git clone the repo, and run it.

    Args:
        git_config ((dict[str, str]): Git configurations used for cloning files,
            including ``repo``, ``branch`` and ``commit``.
        dest_dir (str): The local directory to clone the Git repo into.

    Raises:
        ValueError: If git_config['repo'] is in the wrong format.
        CalledProcessError: If failed to clone git repo.
    """
    is_https = git_config["repo"].startswith("https://")
    is_ssh = git_config["repo"].startswith("git@") or git_config["repo"].startswith("ssh://")
    if not is_https and not is_ssh:
        raise ValueError("Invalid Git url provided.")
    if is_ssh:
        _clone_command_for_ssh(git_config, dest_dir)
    elif "2FA_enabled" in git_config and git_config["2FA_enabled"] is True:
        _clone_command_for_github_like_https_2fa_enabled(git_config, dest_dir)
    else:
        _clone_command_for_github_like_https_2fa_disabled(git_config, dest_dir)


def _clone_command_for_ssh(git_config, dest_dir):
    """Placeholder docstring"""
    if "username" in git_config or "password" in git_config or "token" in git_config:
        warnings.warn("SSH cloning, authentication information in git config will be ignored.")
    _run_clone_command(git_config["repo"], dest_dir)


def _clone_command_for_github_like_https_2fa_disabled(git_config, dest_dir):
    """Placeholder docstring"""
    updated_url = git_config["repo"]
    if "token" in git_config:
        if "username" in git_config or "password" in git_config:
            warnings.warn("Using token for authentication, " "other credentials will be ignored.")
        updated_url = _insert_token_to_repo_url(url=git_config["repo"], token=git_config["token"])
    elif "username" in git_config and "password" in git_config:
        updated_url = _insert_username_and_password_to_repo_url(
            url=git_config["repo"], username=git_config["username"], password=git_config["password"]
        )
    elif "username" in git_config or "password" in git_config:
        warnings.warn("Credentials provided in git config will be ignored.")
    _run_clone_command(updated_url, dest_dir)


def _clone_command_for_github_like_https_2fa_enabled(git_config, dest_dir):
    """Placeholder docstring"""
    updated_url = git_config["repo"]
    if "token" in git_config:
        if "username" in git_config or "password" in git_config:
            warnings.warn("Using token for authentication, " "other credentials will be ignored.")
        updated_url = _insert_token_to_repo_url(url=git_config["repo"], token=git_config["token"])
    _run_clone_command(updated_url, dest_dir)


def _clone_command_for_codecommit(git_config, dest_dir):
    """Check if a git_config param representing a CodeCommit repo is valid.

    If it is, create the command to git clone the repo, and run it.

    Args:
        git_config ((dict[str, str]): Git configurations used for cloning files,
            including ``repo``, ``branch`` and ``commit``.
        dest_dir (str): The local directory to clone the Git repo into.

    Raises:
        ValueError: If git_config['repo'] is in the wrong format.
        CalledProcessError: If failed to clone git repo.
    """
    is_https = git_config["repo"].startswith("https://git-codecommit")
    is_ssh = git_config["repo"].startswith("ssh://git-codecommit")
    if not is_https and not is_ssh:
        raise ValueError("Invalid Git url provided.")
    if "2FA_enabled" in git_config:
        warnings.warn("CodeCommit does not support 2FA, '2FA_enabled' will be ignored.")
    if "token" in git_config:
        warnings.warn("There are no tokens in CodeCommit, the token provided will be ignored.")
    if is_ssh:
        _clone_command_for_ssh(git_config, dest_dir)
    else:
        _clone_command_for_codecommit_https(git_config, dest_dir)


def _clone_command_for_codecommit_https(git_config, dest_dir):
    """Invoke the clone command for codecommit.

    Args:
        git_config: The git configuration.
        dest_dir: The destination directory for the clone.
    """
    updated_url = git_config["repo"]
    if "username" in git_config and "password" in git_config:
        updated_url = _insert_username_and_password_to_repo_url(
            url=git_config["repo"], username=git_config["username"], password=git_config["password"]
        )
    elif "username" in git_config or "password" in git_config:
        warnings.warn("Credentials provided in git config will be ignored.")
    _run_clone_command(updated_url, dest_dir)


def _run_clone_command(repo_url, dest_dir):
    """Run the 'git clone' command with the repo url and the directory to clone the repo into.

    Args:
        repo_url (str): Git repo url to be cloned.
        dest_dir: (str): Local path where the repo should be cloned into.

    Raises:
        CalledProcessError: If failed to clone git repo.
    """
    my_env = os.environ.copy()
    if repo_url.startswith("https://"):
        my_env["GIT_TERMINAL_PROMPT"] = "0"
        subprocess.check_call(["git", "clone", repo_url, dest_dir], env=my_env)
    elif repo_url.startswith("git@") or repo_url.startswith("ssh://"):
        try:
            with tempfile.NamedTemporaryFile() as sshnoprompt:
                with open(sshnoprompt.name, "w") as write_pipe:
                    write_pipe.write("ssh -oBatchMode=yes $@")
                os.chmod(sshnoprompt.name, 0o511)
                my_env["GIT_SSH"] = sshnoprompt.name
                subprocess.check_call(["git", "clone", repo_url, dest_dir], env=my_env)
        except subprocess.CalledProcessError:
            del my_env["GIT_SSH"]
            subprocess.check_call(["git", "clone", repo_url, dest_dir], env=my_env)


def _insert_token_to_repo_url(url, token):
    """Insert the token to the Git repo url, to make a component of the git clone command.

    This method can only be called when repo_url is an https url.

    Args:
        url (str): Git repo url where the token should be inserted into.
        token (str): Token to be inserted.

    Returns:
        str: the component needed fot the git clone command.
    """
    index = len("https://")
    if url.find(token) == index:
        return url
    return url.replace("https://", "https://" + token + "@")


def _insert_username_and_password_to_repo_url(url, username, password):
    """Insert username and password to the Git repo url to make a component of git clone command.

    This method can only be called when repo_url is an https url.

    Args:
        url (str): Git repo url where the token should be inserted into.
        username (str): Username to be inserted.
        password (str): Password to be inserted.

    Returns:
        str: the component needed for the git clone command.
    """
    password = urllib.parse.quote_plus(password)
    # urllib parses ' ' as '+', but what we need is '%20' here
    password = password.replace("+", "%20")
    index = len("https://")
    return url[:index] + username + ":" + password + "@" + url[index:]


def _checkout_branch_and_commit(git_config, dest_dir):
    """Checkout the required branch and commit.

    Args:
        git_config (dict[str, str]): Git configurations used for cloning files,
            including ``repo``, ``branch`` and ``commit``.
        dest_dir (str): the directory where the repo is cloned

    Raises:
        CalledProcessError: If 1. failed to checkout the required branch 2.
            failed to checkout the required commit
    """
    if "branch" in git_config:
        subprocess.check_call(args=["git", "checkout", git_config["branch"]], cwd=str(dest_dir))
    if "commit" in git_config:
        subprocess.check_call(args=["git", "checkout", git_config["commit"]], cwd=str(dest_dir))
