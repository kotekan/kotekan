"""Load configuration files."""
import errno
import json
import os
import subprocess
import sys

yaml = None
try:
    import yaml
except ImportError:
    pass

jinja2 = None
try:
    import jinja2
except ImportError:
    pass


def load_config_file(file_name_full, dump=False, jinja_options=None):
    """
    Load configuration file to json.

    Assumes files are YAML, except when they end with "j2"

    Parameters
    ----------
    file_name_full: str
        Full path to config file.
    dump: bool
        Dump the yaml to stderr (useful for j2 files)
    jinja_options: str
        Add extra jinja variables (JSON format)

    Returns
    -------
    str
        JSON config data
    """
    file_ext = os.path.splitext(file_name_full)[1]
    directory, file_name = os.path.split(file_name_full)

    # Treat all files as pure YAML, unless it is a ".j2" file, then run jinja.
    if file_ext != ".j2":
        # Lint the YAML file, helpful for finding errors
        try:
            output = subprocess.Popen(
                [
                    "yamllint",
                    "-d",
                    "{extends: relaxed, \
                                         rules: {line-length: {max: 100}, \
                                                commas: disable, \
                                                brackets: disable, \
                                                trailing-spaces: {level: warning}}}",
                    file_name_full,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            response, stderr = output.communicate()
            if response != "":
                sys.stderr.write("yamllint warnings/errors for: ")
                sys.stderr.write(str(response))
        # TODO: change to checking for OSError subtypes when Python 2 support is removed
        except OSError as e:
            if e.errno == errno.ENOENT:
                sys.stderr.write("yamllint not installed, skipping pre-validation\n")
            else:
                sys.stderr.write("error with yamllint, skipping pre-validation\n")

        try:
            with open(file_name_full, "r") as stream:
                config_yaml = yaml.safe_load(stream)
        except IOError as err:
            sys.stderr.write("Error reading file " + file_name_full + ": " + str(err))
            sys.exit(-1)
        except yaml.YAMLError as err:
            sys.stderr.write("Error parsing yaml: \n" + str(err) + "\n")
            sys.exit(-1)

        if dump:
            sys.stderr.write(yaml.dump(config_yaml) + "\n")

    else:
        if jinja2 is None:
            raise ImportError("jinja2 is required for processing '.j2' files")

        # Load the template
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(directory), autoescape=jinja2.select_autoescape()
        )
        template = env.get_template(file_name)

        # Parse the optional variables (if any)
        options_dict = {}
        if jinja_options:
            options_dict = json.loads(str(jinja_options))

        # Convert to yaml
        config_yaml_raw = template.render(options_dict)

        # Dump the rendered yaml file if requested
        if dump:
            sys.stderr.write(config_yaml_raw + "\n")

        # TODO Should we also lint the output of the template?
        config_yaml = yaml.safe_load(config_yaml_raw)

    return json.dumps(config_yaml)
