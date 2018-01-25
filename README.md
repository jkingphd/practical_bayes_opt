# A Practical Guide to Bayesian Optimization
This repo is a collection of code associated with the Data Science Nashville meetup held on January 24, 2018: A Practical Guide to Bayesian Optimization. The presentation can be viewed as a powerpoint (**Data_science_nashville_jkk.pptx**) or a pdf (**Data_science_nashville_jkk.pdf**).

- **bayes_opt.yml**: Conda environment necessary to run notebooks.
- **bayes_opt_example.ipynb**: Walkthough of basic Bayesian Optimization technique.
- **gp_example.ipynb**: Visualization of Gaussian process regression in 1D and 2D.
- **gp_kernel.ipynb**: Visualization of different kernels used to fit various target functions.
- **gp_scale.ipynb**: Exploration of scaling issues that can arise when using Gaussian process regression.
- **mercari_modified_utility.py**: Example code showing efficient initialization of Bayesian Optimization object.
- **mercari_prep.ipynb**: Example of Bayesian Optimization used in preprocessing steps on [Mercari Price Prediction Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge).
- **mercari_train.ipynb**: Example of Bayesian Optimization used to train a Catboost Regressor on output from mercari_prep.ipynb.
- **simple_classification.ipynb**: Example of Bayesian Optimization used on [San Francisco Crime Classification](https://www.kaggle.com/c/sf-crime) dataset.
- **simple_regression.ipynb**: Example of Bayesian Optimization used on [House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction) dataset.
- **tuning_strategies.ipynb**: Visualization of traditional tuning strategies.
- **utility_functions.ipynb**: Visualization of different utility functions used in Bayesian Optimization.

# References
## Excellent series of video from Nando de Freitas' course @ UBC.
- [Introduction to Gaussian Processes](https://www.youtube.com/watch?v=4vGiHC35j9s&t=1s)
- [Regression with Gaussian Processes](https://www.youtube.com/watch?v=MfHKW5z-OOA)
- [Bayesian Optimization and Multi-Arm Bandits](https://www.youtube.com/watch?v=vz3D36VXefI)
## A free book on Gaussian Processes.
- [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/)
## A nice publication from Nando de Freitas' group @ UBC.
- [A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning](https://arxiv.org/pdf/1012.2599v1.pdf)
## Scikit-Learn's excellent documentation on Gaussian Processes.
- [Gaussian Processes](http://scikit-learn.org/stable/modules/gaussian_process.html)
