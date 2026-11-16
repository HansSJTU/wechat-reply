import datetime
import sys


def log(*a):
    print(datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3], *a, file=sys.stderr, flush=True)
