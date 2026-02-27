# Dynamic CA-ffe Setup and Installation

Follow these steps to set up the environment, install the code, and run the first test.
Step 1 can be skipped if you wish to use the serial version of the code.

## 1. Install System Dependencies

### Linux (Ubuntu/Debian)

Open a terminal and run:

```bash
sudo apt update
sudo apt install -y build-essential g++ make python3-dev libgomp1 git htop
````
### macOS

Install Homebrew and then install required packages:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install gcc libomp git htop
````

### Windows (MinGW-w64)

Install MinGW-w64 (https://www.mingw-w64.org/getting-started/msys2/) and add g++ to your PATH (cmd approach: set PATH=%PATH%;C:\msys64\ucrt64\bin).

## 2. Clone the GitHub Repository

Install Git (https://github.com/git-guides/install-git) if you skipped the last step.

```bash
git clone https://github.com/MGK-Lab/dyncaffe.git
cd dyncaffe
```

## 3. Create and Activate Conda Environment

Install miniconda (https://www.anaconda.com/docs/getting-started/miniconda/install) and then create the environment as below:

```bash
conda env create -f caffe_conda_env.yml
conda activate caffe
```

## 4. Build the Package

```bash
python setup.py build_ext --inplace
```

## 5. Run the First Test

```bash
python run_caffe.py
```
