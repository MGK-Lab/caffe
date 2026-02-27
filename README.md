# Caffe Setup and Installation

Follow these steps to set up the environment, install the code, and run the first test.

## 1. Install System Dependencies

### Linux (Ubuntu/Debian)

Open a terminal and run:

```bash
sudo apt update
sudo apt install -y build-essential g++ make python3-dev libgomp1 git htop
````
### macOS

Install Homebrew if not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
````

Then install required packages:

```bash
brew install gcc libomp git htop
````

### Windows (MinGW-w64)

1. Install MinGW-w64 and add g++ to your PATH.
2. Ensure Git is installed.
3. Alternatively, install Visual Studio Build Tools with the “Desktop C++” workload.

## 2. Clone the GitHub Repository

```bash
git clone https://github.com/MGK-Lab/dyncaffe.git
cd dyncaffe
```

## 3. Create and Activate Conda Environment

Install miniconda and then creat the environment as below:

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
