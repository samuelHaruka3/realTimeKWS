
# RealTimeKWS

RealTimeKWS (Real-Time Keyword Spotting) is a project that utilizes a Spiking Neural Network (SNN) for voice command recognition in real-time. It's designed to identify specific keywords in spoken language, providing a lightweight and responsive system for voice-controlled applications.The convolutional SNN KWS system we developed excelled in identifying specific keywords, validating its effectiveness real-time audio recognition tasks. The sturdiness of the model was corroborated under diverse noisy environments, ensuring its dependability in actual scenarios. By utilizing PyQt, the GUI delivered a seamless and instinctive user experience, allowing real-time interaction with the audio stream. The design empowers users to effortlessly initiate and deactivate audio processing, observe immediate data visualizations such as audio waveforms, Mel Spectrograms, and spike raster plots, and monitor the model's forecasts as they occur. The figures featured in this segment emphasize the GUI's design and, showcasing its ability to manage frequent updates efficiently without performance degradation.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
- This project has been tested on Linux and Mac systems.

## Installing RealTimeKWS

To install RealTimeKWS, follow these steps:

1. Clone the repository:
```
git clone https://github.com/samuelHaruka3/realTimeKWS.git
```
2.Navigate to the directory where you cloned the repository:
```
cd realTimeKWS
```
3.Create a Conda environment using the environment.yml file included in the repository:
```
conda env create -f environment.yml
```
## Using RealTimeKWS
To use RealTimeKWS, follow these steps:

Activate the newly created Conda environment:
```
conda activate pytorch
```
Run the main script to start the real-time keyword spotting application:
```
python GUIPyQt5.py
```

##Contributors
We appreciate all contributions. If you contribute to this project, please add your name to the list below:

@samuelHaruka3

The spiking jelly package is used.License below.

##Contact
If you want to contact me you can reach me at <20102567d@connect.polyu.hk>.
##License
This project uses the following license: MIT.

## Acknowledgments

This project makes use of the SpikingJelly framework, which is pivotal for the spiking neural network (SNN) implementation. We extend our gratitude to the authors and contributors of SpikingJelly for their significant efforts in advancing spike-based machine intelligence.

## Reference

If you use SpikingJelly for academic research, please cite the following paper:

@article{
doi:10.1126/sciadv.adi1480,
author = {Wei Fang  and Yanqi Chen  and Jianhao Ding  and Zhaofei Yu  and Timothée Masquelier  and Ding Chen  and Liwei Huang  and Huihui Zhou  and Guoqi Li  and Yonghong Tian },
title = {SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence},
journal = {Science Advances},
volume = {9},
number = {40},
pages = {eadi1480},
year = {2023},
doi = {10.1126/sciadv.adi1480},
URL = {https://www.science.org/doi/abs/10.1126/sciadv.adi1480},
eprint = {https://www.science.org/doi/pdf/10.1126/sciadv.adi1480},
abstract = {Spiking neural networks (SNNs) aim to realize brain-inspired intelligence on neuromorphic chips with high energy efficiency by introducing neural dynamics and spike properties. As the emerging spiking deep learning paradigm attracts increasing interest, traditional programming frameworks cannot meet the demands of the automatic differentiation, parallel computation acceleration, and high integration of processing neuromorphic datasets and deployment. In this work, we present the SpikingJelly framework to address the aforementioned dilemma. We contribute a full-stack toolkit for preprocessing neuromorphic datasets, building deep SNNs, optimizing their parameters, and deploying SNNs on neuromorphic chips. Compared to existing methods, the training of deep SNNs can be accelerated 11×, and the superior extensibility and flexibility of SpikingJelly enable users to accelerate custom models at low costs through multilevel inheritance and semiautomatic code generation. SpikingJelly paves the way for synthesizing truly energy-efficient SNN-based machine intelligence systems, which will enrich the ecology of neuromorphic computing. Motivation and introduction of the software framework SpikingJelly for spiking deep learning.}}
![image](https://github.com/samuelHaruka3/realTimeKWS/assets/166841910/5726eadf-cf48-4d80-98bd-a10ed48d6279)
