# python-template

## Visual Studio Code

This repository is optimized for [Visual Studio Code](https://code.visualstudio.com/) which is a great code editor for many languages like Python and Javascript. The [introduction videos](https://code.visualstudio.com/docs/getstarted/introvideos) explain how to work with VS Code. The [Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial) provides an introduction about common topics like code editing, linting, debugging and testing in Python. There is also a [Data Science](https://code.visualstudio.com/docs/datascience/overview) section showing how to work with Jupyter Notebooks and common Machine Learning libraries.

The `.vscode` directory contains configurations for useful extensions like [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens0) and [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python). When opening the repository, VS Code will open a prompt to install the recommended extensions.

## Development Setup

Open the [integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) and run the setup script for your OS. This will install a [Python virtual environment](https://docs.python.org/3/library/venv.html) with all packages specified in `requirements.txt`.

If everything works you should be able to activate the Python environment by entering `source .venv/bin/activate` in the terminal. The command `python src/hello.py` will print the text "hello world" to your terminal.

### Linux and Mac Users

- run the setup script `./setup.sh` or `sh setup.sh`

### Windows Users

- run the setup script `.\setup.ps1`

## Development

- activate python environment: `source .venv/bin/activate`
- run python script: `python <srcfilename.py> `, e.g. `python train.py`
- install new dependency: `pip install sklearn`
- save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`
## Download the dataset from Kaggle
 - When starting the program: Query of the Kaggle user account and the API key

 - Before: Create a Kaggle user account and generate an API key according to the following documentation

 - Link: https://www.kaggle.com/docs/api

 - Getting Started: Installation & Authentication
 The easiest way to interact with Kaggle’s public API is via our command-line tool (CLI) implemented in Python. This section covers installation of the kaggle package and authentication.

 - Installation
 Ensure you have Python and the package manager pip installed. Run the following command to access the Kaggle API using the command line: pip install kaggle (You may need to do pip install --user kaggle on Mac/Linux. This is recommended if problems come up during the installation process.) Follow the authentication steps below and you’ll be able to use the kaggle CLI tool.

 If you run into a kaggle: command not found error, ensure that your python binaries are on your path. You can see where kaggle is installed by doing pip uninstall kaggle and seeing where the binary is. For a local user install on Linux, the default location is ~/.local/bin. On Windows, the default location is $PYTHON_HOME/Scripts.

 - Authentication
 In order to use the Kaggle’s public API, you must first authenticate using an API token. From the site header, click on your user profile picture, then on “My Account” from the dropdown menu. This will take you to your account settings at https://www.kaggle.com/account. Scroll down to the section of the page labelled API:

 To create a new token, click on the “Create New API Token” button. This will download a fresh authentication token onto your machine.

 If you are using the Kaggle CLI tool, the tool will look for this token at ~/.kaggle/kaggle.json on Linux, OSX, and other UNIX-based operating systems, and at C:\Users<Windows-username>.kaggle\kaggle.json on Windows. If the token is not there, an error will be raised. Hence, once you’ve downloaded the token, you should move it from your Downloads folder to this folder.

 If you are using the Kaggle API directly, where you keep the token doesn’t matter, so long as you are able to provide your credentials at runtime.
