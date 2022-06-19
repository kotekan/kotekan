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


def load_config_file(file_name_full, return_dict=False, dump=False, jinja_options=None):
    """
    Load configuration file to json.

    Assumes files are YAML, except when they end with "j2"

    Parameters
    ----------
    file_name_full: str
        Full path to config file.
    return_dict: bool
        Return dict instead of str (default False)
    dump: bool
        Dump the yaml to stderr (useful for j2 files)
    jinja_options: str
        Add extra jinja variables (JSON format)

    Returns
    -------
    str
        JSON config data
    dict
        If return_dict, returns the config as yaml-defined dictionary

    Raises
    ------
    IOError
        If the config file cannot be opened.
    yaml.YAMLError
        If the config file cannot be parsed as yaml.
    ImportError
        If yaml is not installed, or if the file is j2 and jinja2 isn't installed.
    jinja2.TemplateNotFound
        If the file is .j2 and cannot be parsed by jinja2.
    """
    file_ext = os.path.splitext(file_name_full)[1]
    directory, file_name = os.path.split(file_name_full)

    if yaml is None:
        raise ImportError("yaml is required for parsing configuration files.")

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
            raise IOError(
                "Error reading file " + file_name_full + ": " + str(err)
            ) from err
        except yaml.YAMLError as err:
            raise yaml.YAMLError("Error parsing yaml: \n" + str(err) + "\n") from err

        if dump:
            sys.stderr.write(yaml.dump(config_yaml) + "\n")

    else:
        if jinja2 is None:
            raise ImportError("jinja2 is required for processing '.j2' files")

        # Load the template
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(directory),
            autoescape=jinja2.select_autoescape(),
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

    if return_dict:
        return config_yaml

    return json.dumps(config_yaml)


if __name__ == "__main__":
    """
    This script converts YAML or Jinja files into JSON for use by kotekan

    All files without a ".j2" extension are treated as YMAL, files with a ".j2"
    extension are processed by the Jinja template render.

    Extra variables can be passed in with the -e flag, in json/dict format:
     -e '{"my_val": 20, "my_val2": 30, "my_array": [0,2,1]}'

    The JSON is returned in STDOUT, and error messages are returned in STDERR
    """

    parser = argparse.ArgumentParser(
        description="Convert YAML or Jinja files into JSON"
    )
    parser.add_argument("name", help="Config file name", type=str)
    parser.add_argument(
        "-d", "--dump", help="Dump the yaml, useful with .j2 files", action="store_true"
    )
    parser.add_argument(
        "-e", "--variables", help="Add extra jinja variables, JSON format", type=str
    )
    parser.add_argument(
        "-s", "--stderr", help="Print exceptions to stderr", action="store_true"
    )
    args = parser.parse_args()

    options = args.variables

    try:
        print(load_config_file(args.name, dump=args.dump, jinja_options=options))
    except (IOError, yaml.YAMLError, ImportError, jinja2.TemplateNotFound) as err:
        if args.stderr:
            sys.stderr.write(err.message + "\n")
        else:
            raise err
