#!/bin/bash

#pythonw milestone4-report.py --pos NOUN --mode objects --supervised false --objs false --reduction false --plot
#pythonw milestone4-report.py --pos NOUN --mode objects --supervised true --objs false --reduction true --dim 50 --plot
pythonw milestone4-report.py --pos VERB --mode raw --supervised false --objs false --reduction false --plot
pythonw milestone4-report.py --pos VERB --mode objects --supervised false --objs false --reduction false --plot
pythonw milestone4-report.py --pos VERB --mode objects --supervised true --objs false --reduction true --dim 50 --plot
