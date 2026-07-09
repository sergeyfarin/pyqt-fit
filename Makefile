PYTHON=python
EXTRA_ARGS=

.PHONY: coverage test clean

coverage:
	$(PYTHON) -m pytest --cov=pyqt_fit --cov-report=html $(EXTRA_ARGS)

test:
	$(PYTHON) -m pytest $(EXTRA_ARGS)

clean:
	find pyqt_fit -name \*.pyc -exec rm -f '{}' ';'

