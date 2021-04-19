from flask import Flask, json, render_template, request, make_response
from flask_cors import CORS, cross_origin
from requests import get
app = Flask(__name__)
app.config['CORS_ORIGINS'] = ['*']
CORS(app, support_credentials=True)

SITE_NAME = 'http://localhost:12048/'

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def proxy(path):
  print(f'{SITE_NAME}{path}')
  return get(f'{SITE_NAME}{path}').content

@app.route('/get_test')
@cross_origin(origins="*") # allow all origins all methods.
def getBufferData():
    return render_template('get_test.html')

#@app.route('/')
#def hello_world():
#    return 'Hello, World!'

if __name__=="__main__":
    app.run()
