# Installation and contributing

## Install from source

**Axon SDK** can be installed from its [source code](https://github.com/neucom-aps/axon-sdk), open-sourced on Github.

```bash
git clone https://github.com/neucom/axon-sdk.git
cd axon-sdk
pip install -e .
```

**Axon SDK** requires `Python >=3.11` and the following dependencies:
- numpy
- matplotlib
- pytest

The dependencies can be installed using the requirements file:

```bash
cd axon-sdk
pip install -r requirements.txt
```

## Contributing

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

To build a live, locally-hosted HTML version of the docs, use the following commands:

```bash
cd axon-sdk
mdbook serve --open
```

The docs are built automatically as part of our CI/CD pipeline when a new commit arrives.

## Running tests

As part of the merge review process, we’ll check that all the unit tests pass. You can check this yourself (and probably should), by running the unit tests locally.

To run all the unit tests for Rockpool, use pytest:

```bash
cd axon-sdk
pytest tests
```

