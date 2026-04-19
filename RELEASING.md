# Releasing `repralign`

This document is for maintainers who publish `repralign` to PyPI through GitHub Actions and PyPI Trusted Publishing.

## One-Time Setup

1. Push this repository to GitHub.
2. Ensure the default branch is `main`.
3. Confirm the release workflow exists at `.github/workflows/publish.yml`.
4. Log in to PyPI.
5. Open `Publishing` in the PyPI account sidebar.
6. Add a new GitHub Actions publisher with:
   - `PyPI project name`: `repralign`
   - `Owner`: your GitHub username or organization
   - `Repository name`: `repralign`
   - `Workflow name`: `publish.yml`
   - `Environment name`: leave blank unless you later add a GitHub environment

## Before A Release

1. Update `version` in `pyproject.toml`.
2. Run tests locally:

```bash
python3 -m pytest -q
```

3. Optionally build locally:

```bash
python3 -m pip install -e .[dev]
python3 -m build
python3 -m twine check dist/*
```

## Publish A Release

1. Commit and push the release changes to `main`.
2. Create a Git tag that matches the release version, for example `v0.1.1`.
3. Create and publish a GitHub release for that tag.
4. GitHub Actions will run `.github/workflows/publish.yml`.
5. On the first successful upload, PyPI will create the project and attach the trusted publisher.

## Notes

- The package name on PyPI must match `project.name` in `pyproject.toml`.
- The workflow filename registered in PyPI must be `publish.yml`.
- If publishing fails, check the GitHub Actions logs first, then verify the PyPI trusted publisher settings.
