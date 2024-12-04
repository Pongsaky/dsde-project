import logging

print("Initializing logger...")

# Set up logging
logger = logging.getLogger("AFAST-LLMS")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)