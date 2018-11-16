import json
import flask
from rhandler import RHandler


app = flask.Flask(__name__)
rhandler = RHandler()


@app.route("/ping", methods=["GET"])
def ping():
    return flask.Response(status=200)


@app.route("/invocations", methods=["POST"])
def invocations():
    content_type = flask.request.content_type.lower() or "application/json"
    if content_type != "application/json":
        return flask.Response(response="Illegal content type; only application/json accepted")

    accept = flask.request.accept_mimetypes.best_match(["application/json"],
                                                       default="application/json")

    parsed = flask.request.get_json(force=True)

    # TODO: Proper input validation
    if parsed == {}:
        return flask.Response(response="empty payload not expected", status=400)

    forecasts = rhandler.predict(parsed)

    return flask.Response(response=json.dumps(forecasts), status=200, mimetype=accept)