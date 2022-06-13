# Machine Learning Activity Monitoring
This project creates six machine learning models (K-Nearest Neighbors, Extra Trees Classifier, Random Forest Classifier, Bagging Classifier, Support Vector Machine, Convolutional Neural Network) to determine which movements are occuring on an activity monitoring mat developed in the MIT Media Lab.

The user will input two 16x16 pixel datasets 
(row: time series value, col: index of pixel from top-left to bottom-right, values: real number representing how much pressure is on the mat):

    1. 80% Training Dataset           - To train the models and determine the optimal parameters
    2. 20% Testing Dataset            - To test the models and evaluate the most effective model.

This project will output the following:

    1. For each 16x16 activity image, a classification of the activity into seven classes (no activity, left step + right step, left step, right step, planking, push-up down, push-up up).
    1. For each 16x16 yoga image, a classification of the activity into eight classes (no activity, balancing pigeon, dancers pose, eagle dristi, eagle pose, tree pose, warrior, standing).
    2. Precision, Recall, F1 Score, CV Accuracy, CV Standard Deviation, and Runtime for each of the six models.
    3. Optimal hyper-parameters for each of the six models.

## Data
Data is not included in this GitHub repository. Please reach out to me if you would like to request data.

## Author
* **Peter Hwang** - [pghwang](https://github.com/pghwang)

## Acknowledgments
* This project has been created for an undergraduate research project under the MIT Media Lab.
* Special thanks to Irmandy Wicaksono for the guidance and support!

## References
3DKnITS Overview: https://www.media.mit.edu/projects/3dknits/overview/
