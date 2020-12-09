# Contributing

## Set up your dev env

* install virtualenv and virtualenvwrapper globally
* do anything you want to your .zshrc for virtualenv (readthedocs)
* create a virtualenv using the latest stable python.
* enable the virtual env
* install the package deps in dependencies

on ubuntu 18.04, it's all something like this
```
❯ pip3 install virtualenv 

❯ pip3 install virtualenvwrapper

# this is for when you need to make the virtualenv
❯ mkvirtualenv -p /usr/bin/python3.8 your-project

# or, if the virtualenv is already there and you want to use it
❯ workon your-project

# now that we're in our virtualenv, use the virtualenv pip to install the required packages
❯ pip install .

# but wait! we want to be able to run tests, so go ahead and install the test dependencies too
❯ pip install .[test]
```

so after this, your virtualenv is ready to do all the fun stuff in a safe way

## Running basic script

let's execute the command line script to get a pipeline definition from one of the pipeline scripts in the project.

```
❯ workon your-project

❯ get-pipeline-definition --help
usage: Gets the pipeline definition for the pipeline script. [-h] [-n MODULE_NAME] [-kwargs KWARGS]

optional arguments:
  -h, --help            show this help message and exit
  -n MODULE_NAME, --module-name MODULE_NAME
                        The module name of the pipeline to import.
  -kwargs KWARGS, --kwargs KWARGS
                        Dict string of keyword arguments for the pipeline generation (if supported)
```

## Running tests

start up your virtualenv again and let's get to testing

```
❯ workon your-project

❯ python -m pytest   
============================================================= test session starts =============================================================
cachedir: .pytest_cache
plugins: cov-2.10.1
collected 2 items                                                                                                                               

tests/test_pipelines.py::test_that_you_wrote_tests XFAIL                                                                           [ 50%]
tests/test_pipelines.py::test_pipelines_importable PASSED                                                                          [100%]

======================================================== 1 passed, 1 xfailed in 0.04s =========================================================
```

w00t! there you go. have fun developing!
