.PHONY: run test audit all
run:
	python src/project_pipeline.py
test:
	pytest -q
audit:
	python scripts/audit_project.py
all: run audit test
