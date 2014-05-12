"""This module contains internal logging utilities."""

import logging, os

# Logging wrappers
_log = logging.getLogger("instant")
_loghandler = logging.StreamHandler()
_log.addHandler(_loghandler)
#_log.setLevel(logging.WARNING)
_log.setLevel(logging.INFO)
#_log.setLevel(logging.DEBUG)

# Choose method for calling external programs
_default_call_method = 'OS_SYSTEM'
_call_method = os.environ.get("INSTANT_SYSTEM_CALL_METHOD", _default_call_method)
_log.debug('Using call method: %s'%_call_method)

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

if _call_method == 'SUBPROCESS':
    from subprocess import Popen, PIPE, STDOUT

    def get_status_output(cmd, input=None, cwd=None, env=None):
        "Replacement for commands.getstatusoutput which does not work on Windows."
        if isinstance(cmd, str):
            cmd = cmd.strip().split()
        instant_debug("Running: " + str(cmd))
 
        # NOTE: This is not OFED-fork-safe! Check subprocess.py,
        #       http://bugs.python.org/issue1336#msg146685
        #       OFED-fork-safety means that parent should not
        #       touch anything between fork() and exec(),
        #       which is not met in subprocess module. See
        #       https://www.open-mpi.org/faq/?category=openfabrics#ofa-fork
        #       http://www.openfabrics.org/downloads/OFED/release_notes/OFED_3.12_rc1_release_notes#3.03
        pipe = Popen(cmd, shell=False, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

        (output, errout) = pipe.communicate(input=input)
        assert not errout

        status = pipe.returncode
        return (status, output)

elif _call_method == 'OS_SYSTEM':
    import tempfile
    from .paths import get_default_error_dir

    def get_status_output(cmd, input=None, cwd=None, env=None):
        # We don't need function with such a generality.
        # We only need output and return code.
        if not isinstance(cmd, str) or input is not None or \
            cwd is not None or env is not None:
            raise NotImplementedError(
                'This implementation (%s) of get_status_output does'
                ' not accept \'input\', \'cwd\' and \'env\' kwargs.'
                %_call_method)

        f = tempfile.NamedTemporaryFile(dir=get_default_error_dir(),
                                        delete=True)

        # Execute cmd with redirection
        cmd += ' > ' + f.name + ' 2>&1'
        instant_debug("Running: " + str(cmd))
        # NOTE: Possibly OFED-fork-safe, tests needed!
        status = os.system(cmd)

        output = f.read()
        f.close()
        return (status, output)

elif _call_method == 'COMMANDS':
    from commands import getstatusoutput as get_status_output

else:
    instant_error('Incomprehensible environment variable'
                  ' INSTANT_SYSTEM_CALL_METHOD=%s'%_call_method)
