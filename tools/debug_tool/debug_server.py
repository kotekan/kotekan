from flask import Flask, render_template, request, send_file, abort
from flask_cors import CORS
from requests import get
import os

app = Flask(__name__)
app.config["CORS_ORIGINS"] = ["*"]
CORS(app, support_credentials=True)

KOTEKAN_ADDRESS = "http://localhost:12048"

# Display kotekan pipeline
@app.route("/", defaults={"path": ""})
@app.route("/debug_tool/pipeline_tree.html", methods=["GET", "POST"])
def proxy():

    # GET request
    if request.method == "GET":
        data = get(f"{KOTEKAN_ADDRESS}/buffers")
        return render_template("pipeline_tree.html", data=data)

    # POST request
    if request.method == "POST":
        print("Received a POST request")
        print(request.get_json())  # parse as JSON
        return "Sucesss", 200


# Load file
@app.route("/", defaults={"req_path": ""})
@app.route("/<path:req_path>")
def dir_listing(req_path):
    BASE_DIR = "/"

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = os.listdir(abs_path)
    return render_template("files.html", files=files)


# Update kotekan metrics
@app.route("/", defaults={"path": ""})
@app.route("/update", methods=["GET", "POST"])
def update():

    # GET request
    if request.method == "GET":
        data = get(f"{KOTEKAN_ADDRESS}/buffers")
        # print("Data from kotekan: {}".format(data.json()))
        return data.json()

    # POST request
    if request.method == "POST":
        print("Received a POST request")
        print(request.get_json())  # parse as JSON
        return "Sucesss", 200


if __name__ == "__main__":
    app.run()
