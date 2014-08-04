NOSE=nosetests
EXTRA_ARGS=

.PHONY: coverage test

coverage:
	$(NOSE) --with-coverage --cover-html --cover-html-dir=coverage --cover-package=pyqt_fit $(EXTRA_ARGS)

test:
	$(NOSE) $(EXTRA_ARGS)

clean:
	find pyqt_fit -name \*.pyc -exec rm -f '{}' ';'

