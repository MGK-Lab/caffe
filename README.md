# Caffe Setup and Installation

Follow these steps to set up the environment, install the code, and run the first test.

## 1. Install System Dependencies

Open a terminal and run:

```bash
sudo apt update
sudo apt install build-essential git htop
````

## 2. Clone the Repository

```bash
git clone https://github.com/MGK-Lab/dyncaffe.git
cd caffe
```

## 3. Create and Activate Conda Environment

```bash
conda env create -f caffe_conda_env.yml
conda activate caffe
```

## 4. Build the Package

```bash
python setup.py
```

## 5. Run the First Test

```bash
python run_caffe.py
```
