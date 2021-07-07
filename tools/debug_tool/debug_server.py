from flask import Flask, render_template, request, send_file, abort
from flask_cors import CORS
from requests import get
import os

app = Flask(__name__)
app.config["CORS_ORIGINS"] = ["*"]
CORS(app, support_credentials=True)

KOTEKAN_ADDRESS = "http://localhost:12048"

# Load file
@app.route("/", defaults={"req_path": ""})
@app.route("/<path:req_path>")
def dir_listing(req_path):
    BASE_DIR = os.getcwd()

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        file_name = os.path.basename(abs_path)
        if "html" in file_name:
            return render_template(file_name)
        return send_file(abs_path)

    # Show directory contents
    files = os.listdir(abs_path)
    return render_template("files.html", files=files)


# Dynamically read from the given endpoint
@app.route("/<endpoint>", methods=["GET", "POST"])
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

    parser = ArgumentParser()
    parser.add_argument("-a")
    arg = parser.parse_args()
    if arg.a:
        KOTEKAN_ADDRESS = arg.a
    app.run()
