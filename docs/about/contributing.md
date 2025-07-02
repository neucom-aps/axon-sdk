
# Contributing

If you'd like to contribute to **Axon SDK**, then you should begin by forking the public repository at to your own account, commiting some code and opening a PR.

You are also welcome to submit [issues](https://github.com/neucom-aps/axon-sdk/issues) to the Github repo.

We use `main` as a stable branch. You should commit your modifications to a new feature branch.

```bash
$ git checkout -b feature/my-feature develop
...
$ git commit -m 'This is a verbose commit message.'
```

Then push your new branch to your repository

```bash
$ git push -u origin feature/my-feature
```

Use the [Black code formatter](https://github.com/psf/black) on your final commit. This is a requirement. If your modifications aren’t already covered by a unit test, please include a unit test with your merge request. Unit tests use `pytest` and go in the `tests/` directory.

Then when you’re ready, make a merge request on Github the feature branch in your fork to the [Axon SDK repo](https://github.com/neucom-aps/axon-sdk).


## Building the documentation

The documentation is based on **mdbook**.

To build a live, locally-hosted HTML version of the docs, use the following commands (you'll need to install Rust and mdbook):

```bash
cd axon-sdk
mdbook serve --open
```

The docs are built automatically as part of our CI/CD pipeline when a new commit arrives to `main`.

## Running the tests

As part of the merge review process, we’ll check that all the unit tests pass. You can check this yourself (and probably should), by running the unit tests locally.

To run all the unit tests for Axon SDK, use **pytest**:

```bash
cd axon-sdk
pytest tests
```

The test are run automatically as part of our CI/CD pipeline in *every* branch.

