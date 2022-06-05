import argparse
import os
import pathlib
import sys

import joblib
import numpy as np
import sklearn
import yaml
from flask import Flask, jsonify, render_template, request

# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 -p 1234
# params_path = pathlib.Path(__file__).parent.parent.resolve() / "params.yaml"
# sys.argv = [""]
# del sys

webapp_root = "webapp"

params_path = pathlib.Path(__file__).parent.resolve() / "params.yaml"
args = argparse.ArgumentParser()
args.add_argument("--config", default=params_path)
parsed_args = args.parse_args()

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(parsed_args.config)
    model_dir_path = (
        pathlib.Path(__file__).parent.resolve() / config["model_webapp_dir"]
    )
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    return prediction


def form_response(dict_request):
    data = dict_request.values()
    data = [list(map(float, data))]
    response = predict(data)
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req)
                # categories = ['Hernia','Spondylolisthesis','Normal']
                return render_template("index.html", response=response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            error = {"error": e}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
