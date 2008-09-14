
import logging

# Logging wrappers

_log = logging.getLogger("instant")
_loghandler = logging.StreamHandler()
_log.addHandler(_loghandler)
_log.setLevel(logging.WARNING)
_log.setLevel(logging.INFO)
#_log.setLevel(logging.DEBUG)

def get_log_handler():
    return _loghandler

def get_logger():
    return _log

def set_log_handler(handler):
    global _loghandler
    _log.removeHandler(_loghandler)
    _loghandler = handler
    _log.addHandler(_loghandler)

def set_logging_level(level):
    if isinstance(level, str):
        level = level.upper()
        assert level in ("INFO", "WARNING", "ERROR", "DEBUG")
        level = getattr(logging, level)
    else:
        assert isinstance(level, int)
    _log.setLevel(level)

# Aliases for calling log consistently:

def instant_debug(*message):
    _log.debug(*message)

def instant_info(*message):
    _log.info(*message)

def instant_warning(*message):
    _log.warning(*message)

def instant_error(*message):
    _log.error(*message)
    text = message[0] % message[1:]
    raise RuntimeError(text)

def instant_assert(condition, *message):
    if not condition:
        _log.error(*message)
        text = message[0] % message[1:]
        raise AssertionError(text)

# Utility functions for file handling:

def write_file(filename, text):
    "Write text to a file and close it."
    try:
        f = open(filename, "w")
        f.write(text)
        f.close()
    except IOError, e:
        instant_error("Can't open '%s': %s" % (filename, e))


# Taken from http://ivory.idyll.org/blog/mar-07/replacing-commands-with-subprocess
from subprocess import Popen, PIPE, STDOUT
def get_status_output(cmd, input=None, cwd=None, env=None):
    pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

    (output, errout) = pipe.communicate(input=input)
    assert not errout

    status = pipe.returncode

    return (status, output)

