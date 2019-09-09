#!/usr/bin/env python
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

try:
    from future_builtins import *  # noqa  pylint: disable=W0401, W0614
    from future_builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
except ImportError:
    pass
# === End Python 2/3 compatibility

"""
KOTEKAN RELEASE VERSIONING
"""

from pkgutil import get_loader
from subprocess import Popen, PIPE
from pkg_resources import get_distribution

__author__ = "Shiny Brar"
__version__ = "2018.07"
__maintainer__ = "Shiny Brar"
__email__ = "charanjot.brar@mcgill.ca"
__status__ = "Release Candidate"
__developers__ = "Shiny Brar"


def get_version(
    git_dir=None, python_package=None, release_branch="master", debug=False
):
    """
    Returns a version based on git tag,commits,hash and local changes.

    Required Paramters
    ------------------
        None:
            If neither git_dir or python module name are passed, get_version()
            attempts to create a release version based on the git structure of
            the folder of execution of the script.

    Optional Parameters
    -------------------
        git_dir : str
            git folder to use for versioning
            e.g. /home/test/module_name/.git
        python_package : str
            Name of the python module to create a version from
            e.g. import frb_common
                 get_version(frb_common)
        release_branch : str (default is master)
            branch on which the clean release is made.
        debug : boolean (default is false)
            prints extra debug information

    Returns
    -------
        version : string
            Returns a string version with the following format:
                {tag}.{commits_ahead}-{descriptor}
            Where:
                `tag` is the git release tag.
                `commits_ahead` is the number of commits the current checkout
                    is ahead of the git tag.
                `descriptor` is either the git shasum or a 'dirty' flag
                    denoting local uncommited changes.
    e.g.
    2018.07
        Release on master, HEAD@tag and tag found on remote.
    2018.07.0-7y6t5r
        Release on master, HEAD@tag, but tag not found on remote.
    2018.07.4-0i3et6
        Release on master, HEAD was 4 commits ahead of the tag.
    2018.07.14-dirty
        Release on master, HEAD 14 commits ahead and local uncommitted changes.
    develop.2018.07.0-8r8i0e
        Release on develop, HEAD@tag.
        NOTE: A release version `develop.2018.07` is not possible since the
        branch and hash labels are only removed if the release was on master.
    develop.2018.07.54-9uqb40
        Release on develop, HEAD was 54 commits ahead of the tag.
    develop.2018.07.93-dirty
        Release on develop, HEAD 93 commits ahead and there were local
        uncommitted changes.
    """

    # Commands to get git information.
    git_describe_cmd = ["git", "describe", "--tags", "--long", "--dirty"]
    git_remote_cmd = ["git", "ls-remote", "--tags"]
    git_branch_cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]

    # Release formatting.
    release_branch_formatting = "{tag}.{commits_ahead}-{sha}"
    branch_formatting = "{branch}.{tag}.{commits_ahead}-{sha}"
    clean_formatting = "{tag}"

    # Defaults for different types of release states
    dirty = False
    on_release_branch = False
    clean_release = False

    # Defaults for various variables
    branch = None
    git_info = None
    package_dir = None
    git_dir_release = False
    python_package_release = False
    cwd_release = False

    def _run_cmd(cmd, directory="."):
        """
        Runs the output of the cmd as if it was running in the directory.
        Default value for dir is the current working directory.

        Parameters
        ----------
            cmd = list
                Commands to run

        Optional Parameters
        -------------------
            directory : str
                Create the release version based upon the git status of
                provided subdiretory.
        Returns
        -------
            git_info : str
                git information for the passed directory.
        """
        try:
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=directory)
            p.wait()
            p.stderr.close()
            cmd_output = p.stdout.readlines()
        except Exception as e:
            raise e
        return cmd_output

    # Main try/except
    try:
        # Get git information based on git_dir
        if git_dir is not None:
            try:
                assert type(git_dir) == str
                git_info = _run_cmd(cmd=git_describe_cmd, directory=git_dir)
                branch = _run_cmd(cmd=git_branch_cmd, directory=git_dir)

                if (len(git_info) and len(branch)) != 0:
                    # Sanitize git_info and branch
                    branch = branch[0].decode("utf-8").strip()
                    git_info = git_info[0].decode("utf-8").strip()
                    # Create release based on git_dir
                    git_dir_release = True

                if debug:
                    print("Using git_dirt to create release version.")
                    print("git dir: {}".format(git_dir))
                    print("branch : {}".format(branch))
            except Exception:
                raise Exception(
                    "Unable to use git_dir: {} to create release version.".format(
                        git_dir
                    )
                )

        # Get git information based on the python_package
        if python_package is not None and git_dir is None:
            try:
                # Check if the python_package is currently running from a
                # folder under git revision.
                package_dir = get_loader(python_package).filename
                git_info = _run_cmd(cmd=git_describe_cmd, directory=package_dir)
                branch = _run_cmd(cmd=git_branch_cmd, directory=package_dir)
                if debug:
                    print("python package dir      : {}".format(package_dir))
                    print("python package git info : {}".format(git_info))
                    print("python package branch   : {}".format(branch))

                if (len(git_info) and len(branch)) != 0:
                    # Sanitize git_info and branch
                    git_info = git_info[0].decode("utf-8").strip()
                    branch = branch[0].decode("utf-8").strip()
                    python_package_release = True
                else:
                    # If the python module is not running from a git
                    # folder, check if it has version information provided
                    # by setuptools.
                    pkg_version = get_distribution(python_package).version
                    if debug:
                        print("python package setup ver: {}".format(pkg_version))
                    return pkg_version
            except Exception:
                raise Exception(
                    "Unable to use git or setup info to create release version\
                    for python_package: {}".format(
                        python_package
                    )
                )

        # If no git_dir or python_package is provided, use current directory.
        if git_dir is None and python_package is None:
            try:
                git_info = _run_cmd(cmd=git_describe_cmd)
                branch = _run_cmd(cmd=git_branch_cmd)
                if (len(git_info) and len(branch)) != 0:
                    # Sanitize git_info and branch
                    git_info = git_info[0].decode("utf-8").strip()
                    branch = branch[0].decode("utf-8").strip()
                    cwd_release = True
            except Exception:
                raise Exception("Unable to use cwd to create version")

    except Exception as error:
        raise error

    # Create version based on git_info and branch
    try:
        version_parts = git_info.split("-")
        # Assert that the git tag conforms to the CHIME/FRB Versioning Standard
        assert len(version_parts) in (3, 4)
        # Assign tag, commits, and git_sha variables
        tag, commits_ahead, git_sha = version_parts[:3]
        # Find if the commit is dirty
        dirty = len(version_parts) == 4
        if dirty:
            git_sha = "dirty"
        # Check if we are working on the release branch
        # Default is master
        if branch == release_branch:
            on_release_branch = True

        if debug:
            print("tag     : {}".format(tag))
            print("commits : {}".format(commits_ahead))
            print("git_sha : {}".format(git_sha))
            print("dirty   : {}".format(dirty))

        if int(commits_ahead) == 0 and not dirty:
            # Get Remote tag information to verify a clean release version.
            try:
                # Choose the remote tags directory based on the on release.
                if git_dir_release:
                    remote_tags_dir = git_dir
                elif python_package_release:
                    remote_tags_dir = package_dir
                elif cwd_release:
                    remote_tags_dir = "."
                remote_tags = _run_cmd(cmd=git_remote_cmd, directory=remote_tags_dir)

                if debug:
                    print("tags_dir   : {}".format(remote_tags_dir))
                    print("tags       : {}".format(remote_tags))
                    print("git_release: {}".format(git_dir_release))
                    print("pkg_release: {}".format(python_package_release))
                    print("cwd_release: {}".format(cwd_release))

                if len(remote_tags) != 0:
                    for remote_tag in remote_tags:
                        if remote_tag.find(tag) != -1:
                            if debug:
                                print("Matching Remote Tag: {}".format(tag))
                                print("Making a clean tagged release.")
                            clean_release = True
            except Exception:
                print("Unable to fetch remote tags.")

        # Format the release version based on branch and clean status.
        release_format = branch_formatting
        if on_release_branch:
            release_format = release_branch_formatting
        if on_release_branch and clean_release:
            release_format = clean_formatting

        return release_format.format(
            branch=branch, tag=tag, commits_ahead=commits_ahead, sha=git_sha.lstrip("g")
        )
    except Exception:
        raise Exception("Unable to create release version")


if __name__ == "__main__":
    print(get_version())
