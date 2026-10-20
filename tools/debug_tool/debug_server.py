from flask import Flask, render_template, request, send_from_directory, abort
from flask_cors import CORS
from requests import get
from werkzeug.security import safe_join
import os

app = Flask(__name__)
app.config["CORS_ORIGINS"] = ["*"]
CORS(app, support_credentials=True)

KOTEKAN_ADDRESS = "http://localhost:12048"
DUMP_DIR = "./"


@app.route("/")
@app.route("/templates/pipeline_tree.html")
def pipeline_viewer():
    return render_template("pipeline_tree.html")


@app.route("/templates/dump_viewer.html")
def dump_viewer():
    return render_template("dump_viewer.html")


@app.route("/dump_dir", defaults={"req_path": ""})
@app.route("/dump_dir/<path:req_path>")
def dir_listing(req_path):
    base_dir = os.path.realpath(DUMP_DIR)
    abs_path = safe_join(base_dir, req_path)
    if abs_path is None:
        abort(404)

    # Resolve symlinks before checking the dump directory boundary.
    abs_path = os.path.realpath(abs_path)
    if os.path.commonpath([base_dir, abs_path]) != base_dir:
        abort(404)
    if os.path.isfile(abs_path):
        response = send_from_directory(
            base_dir,
            os.path.relpath(abs_path, base_dir),
            mimetype="application/octet-stream",
            as_attachment=True,
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    if not os.path.isdir(abs_path):
        abort(404)

    files = []
    for name in sorted(os.listdir(abs_path)):
        child = os.path.realpath(os.path.join(abs_path, name))
        if os.path.commonpath([base_dir, child]) == base_dir:
            files.append(name)
    return render_template("files.html", files=files)


# Dynamically read from the given endpoint
# "/kotekan_instance" is used to differentiate from file system
@app.route("/kotekan_instance/<endpoint>", methods=["GET", "POST"])
def update(endpoint):

    # GET request
    if request.method == "GET":
        data = get(f"{KOTEKAN_ADDRESS}/{endpoint}")
        # print("Data from kotekan: {}".format(data.json()))
        return data.json()

    # POST request
    if request.method == "POST":
        print("Received a POST request")
        print(request.get_json())  # parse as JSON
        return "Sucesss", 200


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Start Flask server to enable endpoint fetching and file reading."
    )
    parser.add_argument(
        "-a", help="set Kotekan address (default: http://localhost:12048)"
    )
    parser.add_argument("-d", help="set dump file folder (default: ./)")
    arg = parser.parse_args()

    if arg.a:
        KOTEKAN_ADDRESS = arg.a
    if arg.d:
        DUMP_DIR = arg.d

    app.run()
