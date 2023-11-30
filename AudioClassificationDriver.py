import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from AudioFeatureExtractor import AudioFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Feature extraction.  
#Number of segments to split signal
nsegments=1
#number of cepstral coefficients
num_mfcc=20
# Number of samples to use for testing
t_size = 0.5

directory = "./sound_data/mnist/"
audioProcessor = AudioFeatureExtractor(directory)
audioFeatures = audioProcessor.constructMFCCFeatures(nsegments, num_mfcc)
#print(audioFeatures)
print(np.unique(audioFeatures["Target"]))

#Split data into testing and training using stratified sampling
X = audioFeatures.iloc[:,1:]
Y =audioFeatures["Target"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=t_size, random_state=1, stratify=Y)

#Build Classifier
knn = KNeighborsClassifier(n_neighbors=3, weights='distance',)
knn.fit(X_train, Y_train)

#Test perfornmance

#accuracy
print("Training Accuracy:", knn.score(X_train, Y_train))
print("Test Accuracy:", knn.score(X_test, Y_test))

#Confusion matrix
my_labels=['0', '1', '2']
Y_pred = knn.predict(X_test)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(knn, X_test, Y_test,
                                 display_labels=my_labels,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


