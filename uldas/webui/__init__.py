#file: uldas/webui/__init__.py

import threading
import logging
import os

logger = logging.getLogger(__name__)

_config_path = "config/config.yml"
_config_dir = "config"
_scheduler_state = None


def start_webui(config_path: str = "config/config.yml",
                host: str = "0.0.0.0", port: int = 2119,
                scheduler_state=None) -> None:
    """Start the web UI in a daemon thread. Silently skips if Flask is missing."""
    global _config_path, _config_dir, _scheduler_state
    _config_path = config_path
    _config_dir = os.path.dirname(config_path) or "config"
    _scheduler_state = scheduler_state

    try:
        from flask import Flask
    except ImportError:
        logger.info("Flask not installed - web UI disabled. "
                     "Install with: pip install flask")
        return

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    app = Flask(__name__, template_folder=template_dir)

    from uldas.webui.routes import register_routes
    register_routes(app, scheduler_state=scheduler_state)

    def _run():
        wlog = logging.getLogger("werkzeug")
        wlog.setLevel(logging.WARNING)
        app.run(host=host, port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True, name="uldas-webui")
    thread.start()
    print(f"Web UI available at http://localhost:{port}")
    logger.info("Web UI started on %s:%d", host, port)
