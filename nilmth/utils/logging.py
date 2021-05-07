import logging


class CustomFormatter(logging.Formatter):
    err_fmt = "[ERROR] %(msg)s"
    dbg_fmt = "%(msg)s"
    info_fmt = "%(msg)s"
    warn_fmt = "[WARNING] %(msg)s"

    def __init__(self):
        super().__init__(fmt="%(msg)s", datefmt=None, style="%")

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = CustomFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = CustomFormatter.info_fmt

        elif record.levelno == logging.WARNING:
            self._style._fmt = CustomFormatter.warn_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = CustomFormatter.err_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result


logging.basicConfig(format="%(msg)s")
# Set up a logger
logger = logging.getLogger("NILM_logger")
logger.setLevel(logging.DEBUG)

my_formatter = CustomFormatter()

console_handler = logging.StreamHandler()
console_handler.setFormatter(my_formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
