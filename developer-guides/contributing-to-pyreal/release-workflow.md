# Release Workflow

We follow this process to make new releases (at the time, only development leads initiate releases):

1. Create a new branch off of `dev` branch called `release-v*.*.*`.
2. Update the version in `pyreal/__init__.py` and `pyproject.toml` files.
3. Make any final small changes needed, either directly on `release-v*.*.*` or on a feature branch that can then be merged into `release-v*.*.*` with a PR.
4. Make a PR to merge `release-v*.*.*` into `stable` . Add `?template=release_template.md` to the end of the PR URL to enable to the release description template. Fill in the template with a list of changes (ie. PRs) made since the last release.
5. One all checks have passed and reviews are complete, merge the release branch into stable. **Do not delete the release branch**.&#x20;
6. Once merged, tag the merge commit in stable as `v*.*.*`, and push the tag. This will automatically deploy the release to pypi.
7. Merge `release-v*.*.*` back into `dev` with a pull request. Keep the past three release branches; delete any old ones.
8. Make a release on `github.com`, selecting the tag you just made. Follow the title convention of `vX.X.X - YYYY-MM-DD`. Fill in the release notes with the same list of changes you used for the PR description.
