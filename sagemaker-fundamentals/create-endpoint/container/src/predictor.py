# This file implements a flask server for inference. You can modify the file to align with your own inference logic.

import flask
from flask import request

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. 
    In this sample container, we declare
    it healthy if we can load the model successfully."""

    status = 200
    return flask.Response(response='\n', 
        status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def inference():
    """Performed an inference on incoming data. 
    In this sample server, we take data as application/json,
    print it out to confirm that the server received it.  
    """
    content_type = flask.request.content_type
    if flask.request.content_type != "application/json":
        msg = "I just take json, and I am fed with {}".format(
            content_type)
    else:
        msg = "I am fed with json. Therefore, I am happy"

    print("== The entire request object ==")
    print(request)
    
    # define response header 
    header = {} 
    return flask.Response(
        response=msg,
        status=200,
        # header = header,
        mimetype='text/plain')

