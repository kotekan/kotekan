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
  data = get(f'{KOTEKAN_ADDRESS}{url}')
  #print(data.json())
  return render_template('get_test.html', url=url, data=data)

@app.route('/get_test')
@cross_origin(origins="*") # allow all origins all methods.
def getBufferData():
    return render_template('get_test.html')

if __name__=="__main__":
    app.run()
