## Human-Activity-Recognition

The dataset utilized in this project is available from the UCI Machine Learning Repository and consists of sensor data collected from a group of 30 volunteers performing six different activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING). These actiwhile wearing a smartphone with embedded inertial sensors. 

The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.

The independent variables are the data collected from inertial sensor which include the accelerometer and gyroscope 3-axial raw signals tAcc-XYZ and tGyro-XYZ.  The dataset also derives time-domain features (mean, standard deviation) or frequency-domain features (spectral power) from the raw sensor signals. We believe the information capturing the acceleration and angular velocity of the smartphone, time domain features, frequency-features are very important for separating the human activities.

In this project, three supervised machine learning models were used to find an accurate human activity classification. 1D convolutional neutral network (CNN.py) utilized time-domain data collected from accelerometer and gyroscope. Supported vector machine (SVM.py) and Random Forest employed the 561 extracted features detailed in file (HARDataset/features.txt)





