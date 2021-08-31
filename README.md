# README

The project aims at forecasting the electricity price in the long-term (for Italy), while adopting machine learning.

Two solutions are proposed:

1) LSTM (pytorch): the code solves the regression problem using a combination of data cleaning and standardisation
                   techniques as well as filterning (short-term fourier transform). There are a series of .py files
                   into the "codebase/methodology 1" folder. The lack of a large dataset lead to no relevant results.

2) SVR (scikit-learn): the code solves the regression task using the model Support Vector regression. Before training
                       the model, several preprocessing technqiues have been adopted. The Jupyter notebook in
                       "codebase/methodology 2/methodology2.ipynb" contains the full solution.
                       
Both solutions are written in python 3.9
