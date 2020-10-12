all: black lint

black:
	black --line-length 140 cfq

lint:
	flake8 --max-line-length 200 --ignore="E203,E731" cfq
