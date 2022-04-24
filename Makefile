# not sure we really need this
init:
    pip install -r requirements.txt

test:
    py.test tests

.PHONY: init test