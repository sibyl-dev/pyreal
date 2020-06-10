import logging

from flask_restful import Resource
from flask import Flask, request, redirect, url_for

LOGGER = logging.getLogger(__name__)


class Test(Resource):
    def get(self):
        """
        @api {get} /test/ Test get
        @apiGroup Test
        @apiVersion 1.0.0
        """

        return {'message': 'hello get'}, 200

    def post(self):
        """
        @api {post} /test/ Test post
        @apiGroup Test
        @apiVersion 1.0.0
        """

        return {'message': 'hello post'}, 200

    def delete(self):
        """
        @api {delete} /test/ Test delete
        @apiGroup Test
        @apiVersion 1.0.0
        """

        return {'message': 'hello delete'}, 200

    def put(self):
        """
        @api {put} /test/ Test put
        @apiGroup Test
        @apiVersion 1.0.0
        """

        return {'message': 'hello put'}, 200
