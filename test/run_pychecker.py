#!/usr/bin/env python
"Find potential bugs in instant by static code analysis."

# PyChecker skips previously loaded modules
import os, sys, glob, shutil, re, logging, subprocess, hashlib
try:
    import numpy
except:
    pass

import pychecker.checker
import instant

