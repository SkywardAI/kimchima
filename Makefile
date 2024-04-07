#################################################################
TESTDIR:=src/kimchima/tests/

##############################Legacy#############################
.PHONY: setup
setup:
	python setup.py sdist bdist_wheel

.PHONY: upload
upload:
	twine upload dist/* --verbose


################################Poetry################################
.PHONY: poetry
poetry:
	@pipx install poetry==1.8.2


.PHONY: build
build:
	@poetry build


.PHONY: install
install:
	@poetry install -vvv

.PHONY: lint
lint:
	@ruff check --output-format=github .

.PHONY: test
test:
	@poetry run python -m unittest discover ${TESTDIR} -v


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
	@poetry config repositories.source https://pypi.org/project/kimchima



###################################################################################################
# Commit and recommit changes to github
.PONY: commit
commit:
	@echo "Committing changes..."
	@git add .
	@git commit -s -m"${message}"
	@git push origin ${branch}
	@git log -1
	@echo "Changes committed and pushed to github."


.PONY: recommit
recommit:
	@echo "Committing changes..."
	@git add .
	@git commit -s --amend --no-edit
	@git push -f origin ${branch}
	@git log -1
	@echo "Changes committed and pushed to github."