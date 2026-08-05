# Neural Networks I, SMU CS5324/CS7324 
This is a repository for my course at SMU, CS5/7324, previously named machine learning in python. It should be accessible for anyone with coding experience in Python and some machine learning knowledge. Feel free to use any and all code for any purpose. 


The environment should be setup using the following installs for python 3.12 (latest tested version with all code in repository). Different installation instructions are given depending on if you are installing for (1) MacOS (using Apple Silicon) or (2) another OS, as shown below:

## Most OS Install Instructions
For operating systems that do NOT use Apple Silicon, you can use the following commands: 

`conda create --name mlenv3_12 python=3.12`

`conda activate mlenv3_12`

`pip install jupyter numpy scipy pandas matplotlib scikit-learn plotly missingno` 

`pip install protobuf tensorflow-datasets importlib-resources`


## Apple Silicon MacOS Install Instructions
Note: Apple M-series Macs should use a different install procedure as follows:

`python -m pip install tensorflow-macos`

`python -m pip install tensorflow-metal`

`pip install jupyter pandas matplotlib scikit-learn plotly missingno` 

`pip install "numpy<2.0" "scipy<=1.13.1"`

`pip install protobuf tensorflow-datasets importlib-resources`

