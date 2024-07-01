# Development

We recommend to use VSCode and [Devcontainer](../.devcontainer/devcontainer.json) for development. We can keep the development environment consistent and isolated from the host machine. And also same as the CI/CD environment.

## Setup

After open form the Devcontainer, run the following commands to setup the development environment.

```bash
make poetry
```

```bash
make build
```

```bash
# Execution pyhton 3.11 env
make shell
```

After you finish coding, run the follow commands to check the code quality and test.

```bash
make lint
```

```bash
make install
```

```bash
make test
```

