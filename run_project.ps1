$ErrorActionPreference = "Stop"
python -m py_compile src/project_pipeline.py
python src/project_pipeline.py
python scripts/audit_project.py
pytest -q
