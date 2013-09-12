"""This module contains internal logging utilities."""

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
    import inspect
    frame = inspect.currentframe().f_back
    instant_warning("set_logging_level is deprecated but was called "\
                    "from %s, at line %d. Use set_log_level instead." % \
                    (inspect.getfile(frame), frame.f_lineno))
    set_log_level(level)
    
def set_log_level(level):
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
    except IOError as e:
        instant_error("Can't open '%s': %s" % (filename, e))

from subprocess import Popen, PIPE, STDOUT
def get_status_output(cmd, input=None, cwd=None, env=None):
    "Replacement for commands.getstatusoutput which does not work on Windows."
    if isinstance(cmd, str):
        cmd = cmd.strip().split()
    instant_debug("Running: " + str(cmd))
    pipe = Popen(cmd, shell=False, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

    (output, errout) = pipe.communicate(input=input)
    assert not errout

    status = pipe.returncode

    return (status, output)

def get_output(cmd):
    "Replacement for commands.getoutput which does not work on Windows."
    if isinstance(cmd, str):
        cmd = cmd.strip().split()
    pipe = Popen(cmd, shell=False, stdout=PIPE, stderr=STDOUT, bufsize=-1)
    r = pipe.wait()
    output, error = pipe.communicate()
    return output

# Some HPC platforms does not work with the subprocess module and needs commands
#import platform
#if platform.system() == "Windows":
#    # Taken from http://ivory.idyll.org/blog/mar-07/replacing-commands-with-subprocess
#    from subprocess import Popen, PIPE, STDOUT
#    def get_status_output(cmd, input=None, cwd=None, env=None):
#        "Replacement for commands.getstatusoutput which does not work on Windows."
#        pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)
#
#        (output, errout) = pipe.communicate(input=input)
#        assert not errout
#
#        status = pipe.returncode
#
#        return (status, output)
#
#    def get_output(cmd):
#        "Replacement for commands.getoutput which does not work on Windows."
#        pipe = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, bufsize=-1)
#        r = pipe.wait()
#        output, error = pipe.communicate()
#        return output
#
#else:
#    import commands
#    get_status_output = commands.getstatusoutput
#    get_output = commands.getoutput
