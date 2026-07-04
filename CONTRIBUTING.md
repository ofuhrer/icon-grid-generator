# Contributing to ICON Grid Generator

Thanks for helping improve ICON Grid Generator. Contributions are welcome across
the project: bug fixes, new grid variants, diagnostics, NetCDF compatibility,
documentation, tests, examples, and packaging improvements.

## Before You Start

Open an issue or discussion before starting work when a change:

- Adds or changes a public API.
- Changes spherical or planar grid topology, ordering, metrics, refinement
  fields, or NetCDF output contracts.
- Introduces a new dependency.
- Changes release, packaging, or CI behavior.

Small fixes, docs improvements, tests, and clearly scoped refactors can go
straight to a pull request.

## Licensing And Contributor Rights

ICON Grid Generator is distributed under the BSD 3-Clause License. By
contributing to this repository, you agree that your contribution is licensed
under the BSD 3-Clause License unless a file explicitly states otherwise.

All commits must include a Developer Certificate of Origin sign-off. This
certifies that you wrote the contribution or otherwise have the right to submit
it under this project's license. Add the sign-off with:

```bash
git commit -s
```

The resulting commit message must contain a line like:

```text
Signed-off-by: Your Name <you@example.com>
```

Do not submit code, data, generated artifacts, or documentation copied from
another project unless the license permits that reuse and you clearly identify
the source and license in the pull request. Do not submit employer-owned or
third-party work unless you are authorized to contribute it under BSD 3-Clause.

The DCO text is available at <https://developercertificate.org/>.

## Development Setup

Create an isolated Python environment, then install the project in editable
mode with the development extras:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test,docs]"
```

Optional extras are available for NetCDF and xarray workflows:

```bash
python -m pip install -e ".[netcdf,xarray]"
```

## Local Checks

Run these before opening a pull request:

```bash
python -m ruff check .
python -m pytest -q
python -m mkdocs build --strict
```

For packaging, release, README, or metadata changes, also run:

```bash
python -m pip install build twine
rm -rf dist build
python -m build
python -m twine check dist/*
```

## Pull Request Expectations

Each pull request should include:

- A clear summary of what changed and why.
- Tests for behavior changes and bug fixes.
- Documentation updates when user-facing behavior changes.
- DCO-signed commits confirming the contribution can be submitted under
  BSD 3-Clause.
- No unrelated formatting, generated files, or broad refactors.
- Passing CI.

Keep pull requests focused. If a change naturally splits into an API change,
implementation work, documentation, and cleanup, prefer separate PRs when that
makes review easier.

## Domain-Sensitive Changes

Changes to grid generation are easy to make plausible and hard to make correct.
For changes affecting grid math, topology, cell/edge ordering, metric fields,
refinement fields, UUID behavior, limited-area extraction, or NetCDF output,
include the following in the PR description:

- The numerical or scientific rationale for the change.
- References to ICON conventions, upstream behavior, papers, or comparison data
  when available.
- The affected public API, metadata, dimensions, or NetCDF fields.
- Regression tests or comparison evidence showing the intended behavior.
- Any compatibility impact for existing generated grids.

Do not rely on visual inspection alone for geometry or connectivity changes.
Add tests that check counts, bounds, orientation, finite values, parent indices,
or exported variables as appropriate.

## Testing Guidance

Prefer focused tests that prove the contract being changed:

- Public API tests for constructors, options, and exported functions.
- Geometry/topology tests for dimensions, index bounds, adjacency, orientation,
  and finite numeric fields.
- NetCDF tests for variable presence, dimensions, units, and metadata.
- Regression tests for bug fixes, including the smallest grid or region that
  reproduces the problem.

Large generated grids can make CI slow. Use the smallest grid that still covers
the behavior, and reserve larger cases for representative sanity tests.

## Documentation

Update documentation when contributors or users need to know about a change.
The docs live in `docs/` and are built with MkDocs:

```bash
python -m mkdocs build --strict
```

When adding examples, prefer short, runnable snippets that use the public API.

## Style

- Follow the style already present in the codebase.
- Keep public names explicit and domain-specific.
- Validate user-facing inputs close to the public API.
- Avoid introducing dependencies unless they are clearly justified.
- Keep comments short and useful; prefer readable code and focused tests.

## Reporting Issues

When reporting a bug, include:

- The grid spec and options used.
- The Python version and package version or commit.
- The observed error or incorrect output.
- A minimal reproducer.
- Expected behavior.

For numerical or NetCDF contract issues, include relevant dimensions, metadata,
or variable names.

## Release Notes

User-visible changes should update `CHANGELOG.md`. Use concise entries that
explain the impact, not just the implementation detail.
