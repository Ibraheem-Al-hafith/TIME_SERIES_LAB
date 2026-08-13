import os
import logging
import logging.config
import yaml


def setup_logging(config_path: str = "configs/logging.yaml") -> None:
    """
    Initializes the logging infrastructure for the project.

    Loads configuration from a declarative YAML manifest, handles environmental
    safeguards like directory creation, sanitizes runtime contexts such as Jupyter Notebook environments.
    """

    if not os.path.exists(config_path):
        logging.basicConfig(level=logging.INFO)
        logging.warning(
            f"Logging configuration file not found at {config_path}. Fallback to basic configuration."
        )
        return

    try:
        # Loading configuration file
        with open(config_path, "r") as yaml_config_file:
            config = yaml.safe_load(yaml_config_file)

        handlers = config.get("handlers", {})
        for handler_name, handler_config in handlers.items():
            if "filename" in handler_config:
                log_file = handler_config["filename"]
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

        # Clean duplicate handlers to handle notebook environments gracefully
        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

        # Apply structural dictionary configurations:
        logging.config.dictConfig(config=config)
        logging.info(
            f"Logging infrastructure successfully configured via {config_path}."
        )

    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(
            "Fatal failure initializing logging configuration: %s",
            str(e),
            exc_info=True,
        )


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.debug("Hello from loggin.py!!")
