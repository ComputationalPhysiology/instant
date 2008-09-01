
import logging

# Logging wrappers

_log = logging.getLogger("instant")
_loghandler = logging.StreamHandler()
_log.addHandler(_loghandler)

def get_log_handler():
    global _loghandler
    return _loghandler

def get_logger():
    global _log
    return _log

def set_log_handler(handler):
    global _loghandler
    _log.removeHandler(_loghandler)
    _loghandler = handler
    _log.addHandler(_loghandler)

# Aliases for calling log consistently:

def instant_debug(*message):
    global _log
    _log.debug(*message)

def instant_info(*message):
    global _log
    _log.info(*message)

def instant_warning(*message):
    global _log
    _log.warning(*message)

def instant_error(*message):
    global _log
    _log.error(*message)
    text = message[0] % message[1:]
    raise RuntimeError, text

def instant_assert(condition, *message):
    global _log
    if not condition:
        _log.error(*message)
        text = message[0] % message[1:]
        raise RuntimeError, text

# Utility functions for file handling:

def write_file(filename, text):
    "Write text to a file and close it."
    try:
        f = open(filename, "w")
        f.write(text)
        f.close()
    except IOError, e:
        instant_error("Can't open '%s': %s" % (filename, e))


