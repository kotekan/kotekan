from flask import Flask, json, render_template, request, make_response
from flask_cors import CORS, cross_origin
from requests import get
app = Flask(__name__)
app.config['CORS_ORIGINS'] = ['*']
CORS(app, support_credentials=True)

KOTEKAN_ADDRESS = 'http://localhost:12048/'

@app.route('/', defaults={'path': ''})
@app.route('/debug_tool/kotekan_endpoints/<path:url>', methods=["GET", "POST"])
def proxy(url):

  # GET request
  if request.method == 'GET':
    data = get(f'{KOTEKAN_ADDRESS}{url}')
    return render_template('get_test.html', url=url, data=data)

  # POST request
  if request.method == 'POST':
    print("Received a POST request")
    print(request.get_json())  # parse as JSON
    return 'Sucesss', 200

@app.route('/', defaults={'path': ''})
@app.route('/update', methods=["GET", "POST"])
def update():

  # GET request
  if request.method == 'GET':
    data = get(f'{KOTEKAN_ADDRESS}buffers')
    #print("Data from kotekan: {}".format(data.json()))
    return data.json()

  # POST request
  if request.method == 'POST':
    print("Received a POST request")
    print(request.get_json())  # parse as JSON
    return 'Sucesss', 200

if __name__=="__main__":
    app.run()
