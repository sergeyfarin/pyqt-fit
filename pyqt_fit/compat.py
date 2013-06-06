from __future__ import division, print_function, absolute_import
import sys
import csv

PY2 = sys.version_info[0] == 2

if PY2:
    user_text = unicode
    text_type = basestring
    unichr = unichr
    irange = xrange
    lrange = range
    CSV_READ_FLAGS = b"rb"
    DECODE_STRING = lambda s: s.decode('utf_8')
    from itertools import izip

    def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
        # csv.py doesn't do Unicode; encode temporarily as UTF-8:
        csv_reader = csv.reader(unicode_csv_data,
                                dialect=dialect, **kwargs)
        for row in csv_reader:
            # decode UTF-8 back to Unicode, cell by cell:
            yield [unicode(cell, 'utf-8') for cell in row]

    class unicode_csv_writer(object):
        def __init__(self, *args, **kwords):
            self.csv = csv.writer(*args, **kwords)

        def writerows(self, rows):
            rows = [[unicode(val).encode('utf-8') for val in row]
                    for row in rows]
            return self.csv.writerows(rows)

        def writerow(self, row):
            row = [unicode(val).encode('utf-8') for val in row]
            return self.csv.writerow(row)
else:
    user_text = str
    text_type = str
    unichr = chr
    irange = range
    lrange = lambda x: list(range(x))
    CSV_READ_FLAGS = u"rt"
    DECODE_STRING = lambda s: s
    izip = zip
    unicode_csv_reader = csv.reader
    unicode_csv_writer = csv.writer
