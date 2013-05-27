import sys

PY2 = sys.version_info[0] == 2

if not PY2:
    user_text = str
    text_type = str
    unichr = chr
    irange = range
    lrange = lambda x: list(range(x))
    CSV_READ_FLAGS = "rt"
    DECODE_STRING = lambda s: s
    izip = zip
else:
    user_text = unicode
    text_type = basestring
    unichr = unichr
    irange = xrange
    lrange = range
    CSV_READ_FLAGS = "rb"
    DECODE_STRING = lambda s: s.decode('utf_8')
    from itertools import izip

