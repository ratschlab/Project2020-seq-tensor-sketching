#!/usr/bin/env python3

import json

import lib.data as data
from lib.util import *

data.read(file_names_=data.file_names[:2])

seq = data.files[0].seqs[0]
data.compare_exons(seq.id)
