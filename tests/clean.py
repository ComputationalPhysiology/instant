#!/usr/bin/env python

import glob, shutil

for d in glob.glob("test*cache") + glob.glob("*ext"):
    shutil.rmtree(d)
