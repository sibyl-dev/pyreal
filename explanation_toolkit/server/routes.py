import string
from flask import render_template
from flask_restful import Api

import explanation_toolkit.server.resources as ctrl

current_api_version = '/api/v1/'

def add_routes(app):

    # configure RESTful APIs
    api = Api(app)

    api.add_resource(ctrl.test.Test, current_api_version + 'test/')
