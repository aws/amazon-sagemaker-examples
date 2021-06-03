# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new example, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Report Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check [existing open](https://github.com/aws/amazon-sagemaker-examples/issues) and [recently closed](https://github.com/aws/amazon-sagemaker-examples/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aclosed%20) issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps.
* Any modifications you've made relevant to the bug.
* A description of your environment or deployment.


## Contribute via Pull Requests (PRs)

Before sending us a pull request, please ensure that:

* You are working against the latest source on the *master* branch.
* You check the existing open and recently merged pull requests to make sure someone else hasn't already addressed the problem.
* You open an issue to discuss any significant work - we would hate for your time to be wasted.


### Pull Down the Code

1. If you do not already have one, create a GitHub account by following the prompts at [Join Github](https://github.com/join).
1. Create a fork of this repository on GitHub. You should end up with a fork at `https://github.com/<username>/amazon-sagemaker-examples`.
   1. Follow the instructions at [Fork a Repo](https://help.github.com/en/articles/fork-a-repo) to fork a GitHub repository.
1. Clone your fork of the repository: `git clone https://github.com/<username>/amazon-sagemaker-examples` where `<username>` is your github username.


### Run the Linters

1. Install tox using `pip install tox`
1. cd into the amazon-sagemaker-examples folder: `cd amazon-sagemaker-examples` or `cd /environment/amazon-sagemaker-examples`
1. Run the following tox command and verify that all linters pass: `tox -e black-check,black-nb-check`
1. If the linters did not pass, run the following tox command to fix the issues: `tox -e black-format,black-nb-format`


### Test Your Notebook End-to-End

Our [CI system](https://github.com/aws/sagemaker-example-notebooks-testing) runs modified or added notebooks, in parallel, for every Pull Request.
Please ensure that your notebook runs end-to-end so that it passes our CI.

The `sagemaker-bot` will comment on your PR with a link for `Build logs`.
If your PR does not pass CI, you can view the logs to understand how to fix your notebook(s) and code.

### Commit Your Change

Use imperative style and keep things concise but informative. See [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) for guidance.

### Send a Pull Request

GitHub provides additional document on [Creating a Pull Request](https://help.github.com/articles/creating-a-pull-request/).


Please remember to:
* Send us a pull request, answering any default questions in the pull request interface.
* Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.


## Example Notebook Best Practices

Here are some general guidelines to follow when writing example notebooks:
* Use the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) wherever possible, rather than `boto3`.
* Do not hardcode information like security groups, subnets, regions, etc.
    ```python
    # Good
    loader = botocore.loaders.create_loader()
    resolver = botocore.regions.EndpointResolver(loader.load_data("endpoints"))
    resolver.construct_endpoint("s3", region)

    # Bad
    cn_regions = ['cn-north-1', 'cn-northwest-1']
    region = boto3.Session().region_name
    endpoint_domain = 'com.cn' if region in cn_regions else 'com'
    's3.{}.amazonaws.{}'.format(region, endpoint_domain)
    ```
* Do not require user input to run the notebook.
  * 👍 `bucket = session.default_bucket()`
  * 👎 `bucket = <YOUR_BUCKET_NAME_HERE>`
* Lint your code and notebooks. (See the section on [running the linters](#run-the-linters) for guidance.)
* Use present tense.
  * 👍 "The estimator fits a model."
  * 👎 "The estimator will fit a model."
* When referring to an AWS product, use its full name in the first invocation.
  (This applies only to prose; use what makes sense when it comes to writing code, etc.)
  * 👍 "Amazon S3"
  * 👎 "s3"
* Provide links to other ReadTheDocs pages, AWS documentation, etc. when helpful.
  Try to not duplicate documentation when you can reference it instead.
  * Use meaningful text in a link.
    * 👍 You can learn more about [hyperparameter tuning with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html) in the SageMaker docs.
    * 👎 Read more about it [here](#).


## Find Contributions to Work On

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels ((enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any ['help wanted'](https://github.com/aws/amazon-sagemaker-examples/labels/help%20wanted) issues is a great place to start.


## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security Issue Notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](https://github.com/aws/amazon-sagemaker-examples/blob/master/LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
