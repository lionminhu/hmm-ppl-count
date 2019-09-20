# Modeling People Count in a Building with Hidden Markov Model

Final project for Artificial Intelligence course ([course link](https://bi.snu.ac.kr/Courses/4ai19s/4ai19s.html)) that uses Hidden Markov Models to model and predict the count of people into and out of a building according to CalIt2 data set.

This repository is run at following specifications

- Linux Ubuntu 16.04
- Python 3.6
- Numpy 1.16.3

Hidden Markov Model implementation is a slight modification of the library code from `hidden_markov` package ([link](https://pypi.org/project/hidden_markov/)).

To run this repository, first download the data set from the UCI Machine Learning Repository ([link](https://archive.ics.uci.edu/ml/datasets/CalIt2+Building+People+Counts)). Make sure that the `CalIt2.events` and `Calit2.data` files are in the root of this repository. Afterwards, run `python main.py`.

## Documentation
Poster used for presentation can be checked [here](https://github.com/lionminhu/ai-project-private/blob/master/docs/poster.pdf).

For details not covered in the poster, please check the [report](https://github.com/lionminhu/ai-project-private/blob/master/docs/report.pdf).
