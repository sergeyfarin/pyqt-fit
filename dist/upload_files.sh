#!/usr/bin/env bash

USERNAME="$1"
VERSION="$2"
COMMENT="$3"

./googlecode_upload.py -p pyqt-fit -u $USERNAME -l Featured,Type-Package,OpSys-All,Python2 -s "Egg package for Python 2.7. $COMMENT" PyQt_Fit-$VERSION-py2.7.egg

./googlecode_upload.py -p pyqt-fit -u $USERNAME -l Featured,Type-Package,OpSys-All,Python3 -s "Egg package for Python 3.2. $COMMENT" PyQt_Fit-$VERSION-py3.2.egg

./googlecode_upload.py -p pyqt-fit -u $USERNAME -l Featured,Type-Package,OpSys-All -s "PIP package for python 2 and 3. $COMMENT" PyQt-Fit-$VERSION.tar.gz

