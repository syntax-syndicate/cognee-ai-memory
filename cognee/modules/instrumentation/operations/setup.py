import logging
import os


def setup_instrumentation():
    instrumentation_api_key = os.getenv("INSTRUMENTATION_API_KEY")
    if instrumentation_api_key:
        import logfire
        logfire.configure(token=instrumentation_api_key)
        logfire.instrument_system_metrics(base="full")

        env = (os.getenv("ENV", "DEV")).lower()

        logging_level = logging.ERROR if env == "prod" else logging.INFO
        logging.basicConfig(
            level=logging_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logfire.LogfireLoggingHandler(level=logging_level)],
        )
