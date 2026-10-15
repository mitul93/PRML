# Pattern Recognition and Machine Learning

This repository contains my personal notes and experiments while studying *Pattern Recognition and Machine Learning* by Christopher M. Bishop.

I am using Jupyter notebooks to explore the concepts presented in the book. Notebooks provide a convenient way to combine different elements of the learning process in one place.

- **Explanations:** Markdown makes it easy to write notes and organize ideas.
- **Mathematics:** LaTeX support allows me to write mathematical notation and formulas.
- **Visualizations:** Python makes it easy to create plots and visualize mathematical concepts.

The goal is to develop a deeper understanding of the material by combining theory, mathematical derivations, visualizations, and practical experiments.

## Disclaimer

All credit for the original work belongs to Christopher M. Bishop, the author of *Pattern Recognition and Machine Learning*.

This repository is an independent, personal learning project. The notes and implementations are created for educational purposes, and I do not claim ownership of the original material or the ideas presented in the book.

The intention is simply to document my learning journey and improve my understanding of the subject.

## Features

- Works with VS Code Dev Containers
- Works with VS Code Jupyter integration
- Supports Docker and Podman
- Non-root user inside container
- UID/GID matching with host
- Forwards host SSH keys for Git access

---

## Requirements

- VS Code
- Dev Containers extension
- Docker or Podman installed on Host

---

## Prerequisite

### Using with Podman

#### Install and verify podman

Make sure that podman is installed and accessible
```bash
$ podman --version
```
#### Configure VS Code to use Podman

Open VS Code settings for workspace
- `Ctrl + Shift + P` -> Preference Open Workspace Settings
- Search `containers.containerClient` and set it to `podman`. This tells the VS Code Dev Containers extension to use Podman instead of Docker.

### Using with Docker

TODO : Test and Write

## Open project in VS Code

Clone this repository (if you haven’t already) and open it in VS Code.

## Start Devcontainer

Press Ctrl + Shift + P → Dev Containers: Reopen in Container

## Run hello world inside devcontainer

```
python example/hello_world.py 
```
or

Open file [hello_world.ipynb](example_notebook/hello_world.ipynb) and click on `Run All` button.