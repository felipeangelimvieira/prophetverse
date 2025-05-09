Contributing to Prophetverse

## Finding an issue to contribute to

If you are brand new to Prophetverse or open-source development, we recommend searching the [GitHub “issues” tab](https://github.com/felipeangelimvieira/prophetverse/issues) to find issues that interest you. Unassigned issues labeled Docs and good first issue are typically good for newer contributors.

Once you’ve found an interesting issue, it’s a good idea to assign the issue to yourself, so nobody else duplicates the work on it. On the Github issue, a comment with the exact text take to automatically assign you the issue (this will take seconds and may require refreshing the page to see it).

If for whatever reason you are not able to continue working with the issue, please unassign it, so other people know it’s available again. You can check the list of assigned issues, since people may not be working in them anymore. If you want to work on one that is assigned, feel free to kindly ask the current assignee if you can take it (please allow at least a week of inactivity before considering work in the issue discontinued). To submit your contribution make a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

## Tips for a successful pull request

If you have made it to the Making a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) phase, one of the core contributors may take a look.

## What is a good pull request?

- Reference an open issue for non-trivial changes to clarify the PR’s purpose

- Ensure you have appropriate tests. These should be the first part of any PR

- Keep your pull requests as simple as possible. Larger PRs take longer to review

- Ensure that CI is in a green state. Reviewers may not even look otherwise

- Keep Updating your pull request, either by request or every few days

## Creating a development environment

1. Step 1: [install python](https://wiki.python.org/moin/BeginnersGuide)

2. Step 1: [install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
3. Step 2: install dependencies (including `dev` dependencies) with `uv`:
    <br> ```uv sync``` </br>

## Contributing to the documentation

Contributing to the documentation benefits everyone who uses Prophetverse. We encourage you to help us improve the documentation, and you don’t have to be an expert on Prophetverse to do so! In fact, there are sections of the docs that are worse off after being written by experts. If something in the docs doesn’t make sense to you, updating the relevant section after you figure it out is a great way to ensure it will help the next person.

## About the documentation

The documentation is written in [mkdocs](https://www.mkdocs.org/), you can learn more at [mkdocs getting started guide](https://www.mkdocs.org/getting-started/).

## Contributor community

Community slack: None yet.

## Contributing to the code base

### Code standards

Writing good code is not just about what you write. It is also about how you write it. During Continuous Integration testing, several tools will be run to check your code for stylistic errors. Generating any warnings will cause the test to fail. Thus, good style is a requirement for submitting code to Prophetverse.There are of tools in Prophetverse to help contributors verify their changes before contributing to the project

#### [Pytest](https://docs.pytest.org/en/7.1.x/contents.html)

You can test your code with pytest integration with this `uv` command
<br> ```uv run pytest```

The CI tests are computationally intensive, so if you want to do a faster test you can run a [smoke test](https://en.wikipedia.org/wiki/Smoke_testing_(software)) with the command
<br> ```uv run pytest -m "not ci"```

If you also wanna run the tests even faster feel free to parallel processing the tests with [pytest-xdist](https://pytest-xdist.readthedocs.io/en/latest/how-to.html#making-session-scoped-fixtures-execute-only-once).

#### [Pre-commit](https://pre-commit.com/)

Additionally, Continuous Integration will run code formatting checks like black, isort, and mypy and more using pre-commit hooks. Any warnings from these checks will cause the Continuous Integration to fail; therefore, it is helpful to run the check yourself before submitting code. This can be done by installing pre-commit (which should already have happened if you followed the instructions in Setting up your development environment) and then running:

##### pre-commit install

from the root of the Prophetverse repository. Now all of the styling checks will be run each time you commit changes without your needing to run each one manually. In addition, using pre-commit will also allow you to more easily remain up-to-date with our code checks as they change.

##### pre-commit usage

Note that if needed, you can skip these checks with git commit --no-verify.

If you don’t want to use pre-commit as part of your workflow, you can still use it to run its checks with one of the following:

<br> ``` pre-commit run --files <files you have modified> ``` </br>
<br> ``` pre-commit run --from-ref=upstream/main --to-ref=HEAD --all-files ``` </br>

without needing to have done pre-commit install beforehand.

Finally, we also have some slow pre-commit checks, which don’t run on each commit but which do run during continuous integration. You can trigger them manually with:

<br> ``` pre-commit run --hook-stage manual --all-files ``` </br>

#### Conventional commits

Try to use conventional commits. Each type is grouped into a GitHub release section with the same name, and each type increments a part of the version number. The types are:

| Type       | Description                                                                                                     | Version Increment | Example                                   |
| ---------- | --------------------------------------------------------------------------------------------------------------- | ----------------- | ----------------------------------------- |
| `feat`     | Added a new feature                                                                                             | `minor`           | `feat: add new feature`                   |
| `fix`      | Fixed a bug                                                                                                     | `patch`           | `fix: fix bug`                            |
| `perf`     | Improved code performance                                                                                       | `patch`           | `perf: improve performance`               |
| `refactor` | A code change that neither fixes a bug nor adds a feature                                                       | `patch`           | `refactor: refactor code`                 |
| `docs`     | Changes only to documentation                                                                                   | `patch`           | `docs: update documentation`              |
| `style`    | Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc.)         | `patch`           | `style: fix style`                        |
| `test`     | Add missing tests or correct existing tests                                                                     | `patch`           | `test: add test`                          |
| `chore`    | Changes to the build process or auxiliary tools and libraries such as documentation generation                  | `patch`           | `chore: update build process`             |
| `ci`       | Changes to our CI configuration files and scripts                                                               | `patch`           | `ci: update CI`                           |
