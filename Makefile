.PHONY: setup
setup:
	python setup.py sdist bdist_wheel

.PHONY: upload
upload:
	twine upload dist/* --verbose


.PHONY: test
test:
	@python -m unittest -v


################################poetry################################
.PHONY: poetry
poetry:
	@pipx install poetry==1.8.2


.PHONY: build
build:
	@poetry build


.PHONY: install
install:
	@poetry install


# build and publish
.PHONY: publish
publish:
	@poetry publish --build

# list current configuration
.PHONY: config
config:
	@poetry config --list

.PHONY: source
source:
	@poetry config repositories.source https://pypi.org/simple