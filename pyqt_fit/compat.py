import sys

PY2 = sys.version_info[0] == 2

if not PY2:
    text_type = basestring
    string_types = (str,)
    unichr = chr
    irange = range
    lrange = lambda x: list(range(x))
    CSV_READ_FLAGS = "rt"
    DECODE_STRING = lambda s: s
    izip = zip
    basestring = str
else:
    text_type = unicode
    string_types = (str, unicode)
    unichr = unichr
    irange = xrange
    lrange = range
    CSV_READ_FLAGS = "rb"
    DECODE_STRING = lambda s: s.decode('utf_8')
    from itertools import izip

