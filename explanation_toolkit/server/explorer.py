import logging
import os
import sys
import ssl

from flask import Flask
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from termcolor import colored

from explanation_toolkit.server import g
from explanation_toolkit.server.routes import add_routes
from explanation_toolkit.server.utils import import_object

LOGGER = logging.getLogger(__name__)


class Explorer:

    def __init__(self, conf):
        self._conf = conf.copy()

    def _init_flask_app(self, env):
        app = Flask(
            __name__,
            static_url_path=''
        )

        app.config.from_mapping(**self._conf)

        if env == 'production':
            app.config.from_mapping(DEBUG=False, TESTING=False)

        elif env == 'development':
            app.config.from_mapping(DEBUG=True, TESTING=True)

        elif env == 'test':
            app.config.from_mapping(DEBUG=False, TESTING=True)

        CORS(app)
        add_routes(app)

        # set up global variables
        g['config'] = self._conf
        g['app'] = app

        return app


    def run_server(self, env, port):

        env = self._conf['ENV'] if env is None else env
        port = self._conf['server_port'] if port is None else port

        # env validation
        if env not in ['development', 'production', 'test']:
            LOGGER.exception("env '%s' is not in "
                             "['development', 'production', 'test']", env)
            raise ValueError

        # in case running app with the absolute path
        sys.path.append(os.path.dirname(__file__))

        app = self._init_flask_app(env)

        LOGGER.info(colored('Starting up FLASK APP in {} mode'.format(env),
                            'yellow'))

        LOGGER.info(colored('APIs are available on:', 'yellow')
                    + '  http://localhost:' + colored(port, 'green') + '/')

        if env == 'development':
            app.run(debug=True, port=port)
            # app.run(debug=True, port=port, ssl_context="adhoc")

        elif env == 'production':
            server = WSGIServer(('0.0.0.0', port), app, log=None)
            # server = WSGIServer(('0.0.0.0', port), app, ssl_context="adhoc", log=None)
            server.serve_forever()
