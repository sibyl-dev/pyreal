# Release Workflow

We follow this process to make new releases (at the time, only development leads initiate releases):

1. Create a new branch off of `dev` branch called `release-v.*.*.*`.
2. Update the version in `pyreal/__init__.py` and `pyproject.toml` files.
3. Make any final small changes needed, either directly on `release-v.*.*.*` or on a feature branch that can then be merged into `release-v.*.*.*` with a PR.
4. Make a PR to merge `release-v.*.*.*` into `stable`
5. Once merged, tag the merge commit in stable, and push the tag. This will automatically deploy the release to pypi.
6. Merge `release-v.*.*.*` back into `dev` with a pull request
7. Make a release on github.com, filling in the release notes with a list of pull requests made since the last release.