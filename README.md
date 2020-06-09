# GPGPU Benchmark Tooling

## Usage

A config needs:

* `create` step: copy or clone project
* `setup` step: install package, build project, ...
* `run` step: run executable

Create a config as follow:

```yaml
# Cuda Base Line Test Benchmark

create:
    name: "{benchmark name}"
    repository: "{repository}"
    tag: "{tag or branch}"

setup:
    steps:
        - enter: FOLDER
        - cmd: mkdir build
        - enter: build
        - cmd: cmake ..
        - cmd: cmake --build . --config Release
run:
    exec: {binary}
    args: ""
    steps:
        - enter: FOLDER
        - enter: build/Release
        - runcmd: ''
```

## Local benchmark (not using git)

The only difference is in the create step:

```yaml
create:
    name: "{benchmark name}"
    local: "{path/to/project/}"
```

## Keywords:

### Create

* `create/name`: name of the benchmark
* `create/repository`: repository
* `create/tag`: tag or branch (opt, default master)
* `create/local`: path to local directory (overwrite repository and tag)

### Setup

* `setup/steps`: list of steps to setup project
* `setup/steps/enter`: update working directory
* `setup/steps/cmd`: run command

### Run

* `run/exec`: executable
* `run/args`: args that will be passed to executable
* `run/steps`: list of steps to run
* `run/steps/enter`: update working directory
* `run/steps/cmd`: run command
* `run/steps/runcmd`: will run the executable (if given value: "{value} {exec} {args}" ex: python exec.py --help)


## Disclosure

It is a very early version, there is no check on validity and command executed.
It is recommended to run it in a docker container if not sure.
`rm` type command will be executed.