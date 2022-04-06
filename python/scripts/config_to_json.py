#!/usr/bin/env python
"""
This script converts YAML or Jinja files into JSON for use by kotekan

All files without a ".j2" extension are treated as YMAL, files with a ".j2"
extension are processed by the Jinja template render.

Extra variables can be passed in with the -e flag, in json/dict format:
 -e '{"my_val": 20, "my_val2": 30, "my_array": [0,2,1]}'
   
The JSON is returned in STDOUT, and error messages are returned in STDERR
"""
try:
    import yaml, json, sys, os, subprocess, errno, argparse
except ImportError as err:
    sys.stderr.write(
        "Missing python packages, run: pip3 install -r python/requirements.txt\n"
        + "Error message: "
        + str(err)
        + "\n"
    )
    exit(-1)

# Setup arg parser
parser = argparse.ArgumentParser(description="Convert YAML or Jinja files into JSON")
parser.add_argument("name", help="Config file name", type=str)
parser.add_argument(
    "-d", "--dump", help="Dump the yaml, useful with .j2 files", action="store_true"
)
parser.add_argument(
    "-e", "--variables", help="Add extra jinja variables, JSON format", type=str
)
args = parser.parse_args()

options = args.variables

# Split the file name into the name, directory path, and extension
file_name_full = args.name
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

    if args.dump:
        sys.stderr.write(yaml.dump(config_yaml) + "\n")

    sys.stdout.write(json.dumps(config_yaml))

else:
    try:
        from jinja2 import FileSystemLoader, Environment, select_autoescape
        from jinja2 import TemplateNotFound
    except ImportError as err:
        sys.stderr.write(
            "Jinja2 required for '.j2' files, run pip3 install -r python/requirements.txt"
            + "\nError message: "
            + str(err)
            + "\n"
        )
        exit(-1)

    # Load the template
    env = Environment(
        loader=FileSystemLoader(directory), autoescape=select_autoescape()
    )
    try:
        template = env.get_template(file_name)
    except TemplateNotFound as err:
        sys.stderr.write("Could not open the file: " + file_name_full + "\n")
        exit(-1)

    # Parse the optional variables (if any)
    options_dict = {}
    if options:
        options_dict = json.loads(str(options))

    # Convert to yaml
    config_yaml_raw = template.render(options_dict)

    # Dump the rendered yaml file if requested
    if args.dump:
        sys.stderr.write(config_yaml_raw + "\n")

    # TODO Should we also lint the output of the template?
    try:
        config_yaml = yaml.safe_load(config_yaml_raw)
    except yaml.YAMLError as err:
        sys.stderr.write("Error parsing yaml: \n" + str(err) + "\n")
        sys.exit(-1)

    sys.stdout.write(json.dumps(config_yaml))
