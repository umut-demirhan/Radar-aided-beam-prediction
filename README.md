# Radar Aided 6G Beam Prediction: Deep Learning Algorithms and Real-World Demonstration
This is a python code package related to the following article:
Umut Demirhan, and Ahmed Alkhateeb, "[Radar Aided 6G Beam Prediction: Deep Learning Algorithms and Real-World Demonstration](https://ieeexplore.ieee.org/document/9771564),", in 2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2655-2660

# Instructions to Reproduce the Results 
The scripts for generating the results of the ML solutions in the paper. This script adopts Scenario 9 of DeepSense6G dataset.

**To reproduce the results, please follow these steps:**
1. Download [the radar aided beam prediction dataset of DeepSense 6G/Scenario 9](https://deepsense6g.net/radar-aided-beam-prediction/).
2. Download (or clone) the repository into a directory.
3. Extract the dataset into the repository directory 
   (If needed, the dataset directory can be changed at line 28 of scenario9_radar_beam-prediction_inference.py)
4. Comment in one of the lines 21-23 of scenario9_radar_beam-prediction_inference.py.
   Activating the lines allow selecting the radar-cube, range-velocity and range-angle ML solutions, respectively.
5. Run scenario9_radar_beam-prediction_inference.py file.

**Results of the script**
| Solution       | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 |
| :------------- | ----- | ----- | ----- | ----- | ----- |
| Radar Cube     | 41.65 | 60.88 | 74.37 | 87.18 | 91.91 |
| Range Velocity | 42.33 | 60.88 | 74.20 | 83.81 | 89.54 |
| Range Angle    | 45.70 | 65.60 | 79.60 | 88.36 | 93.25 |

If you have any questions regarding the code and used dataset, please write to DeepSense 6G dataset forum https://deepsense6g.net/forum/ or contact [Umut Demirhan](mailto:udemirhan@asu.edu?subject=[GitHub]%20Beam%20prediction%20implementation).

# Training Models
**Note: The newly trained models do not exactly reproduce the results. It is provided as a reference.**

1. Download [the radar aided beam prediction dataset of DeepSense 6G/Scenario 9](https://deepsense6g.net/radar-aided-beam-prediction/).
2. Download (or clone) the repository into a directory.
3. Extract the dataset into the repository directory 
   (If needed, the dataset directory can be changed at line 18 of train.py)
4. Set the parameters. The parameter data_type (line 22) determines the radar-cube, range-velocity and range-angle ML solutions.
5. Run train.py file.

# Abstract of the Article
Adjusting the narrow beams at millimeter wave (mmWave) and terahertz (THz) MIMO communication systems is associated with high beam training overhead, which makes it hard for these systems to support highly-mobile applications. This overhead can potentially be reduced or eliminated if sufficient awareness about the transmitter/receiver locations and the surrounding environment is available. In this paper, efficient deep learning solutions that leverage radar sensory data are developed to guide the mmWave beam prediction and significantly reduce the beam training overhead. Our solutions integrate radar signal processing approaches to extract the relevant features for the learning models, and hence optimize their complexity and inference time. The proposed machine learning based radar-aided beam prediction solutions are evaluated using a large-scale real-world mmWave radar/communication dataset and their capabilities were demonstrated in a realistic vehicular communication scenario. In addition to completely eliminating the radar/communication calibration overhead, the proposed algorithms are able to achieve around 90% top-5 beam prediction accuracy while saving 93% of the beam training overhead. This highlights a promising direction for addressing the training overhead challenge in mmWave/THz communication systems.

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
If you in any way use this code for research that results in publications, please cite our original article:
> U. Demirhan and A. Alkhateeb, "[Radar Aided 6G Beam Prediction: Deep Learning Algorithms and Real-World Demonstration](https://ieeexplore.ieee.org/document/9771564)," 2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2655-2660, doi: 10.1109/WCNC51071.2022.9771564.

If you use the [DeepSense 6G dataset](www.deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” to be available on arXiv, 2022. [Online]. Available: https://www.DeepSense6G.net
