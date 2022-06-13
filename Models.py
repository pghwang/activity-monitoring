 import pandas as pd
from pandas import read_csv
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import preprocessing
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import timeit

filename = "training.csv"
test_filename = "testing.csv"

def load_dataset(filename):
    # Load Dataset
    dataset = pd.read_csv(filename)
    dataset = dataset.values
    # Split Dataset Into Training and Testing Datasets
    X, y = dataset[:,1:], dataset[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # Scale Data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def load_dataset_cv(filename):
    # Load Dataset
    data = read_csv(filename)
    # Parse as Numeric Values
    data = data.values
    # Split Dataset into X and y
    X, y = data[:,1:], data[:,0]
    # Label Encode y (0 or 1)
    y = LabelEncoder().fit_transform(y)
    return X, y

# Load Dataset
def load_dataset_cnn(filename):
    dataframe = pd.read_csv(filename)
    dataset = dataframe.values
    X, y = dataset[:,1:].astype(float), dataset[:,0]
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.20)
    encoder = LabelEncoder()
    encoder.fit(trainy)
    trainy = encoder.transform(trainy)
    encoder.fit(testy)
    testy = encoder.transform(testy)
    scaler = StandardScaler()
    scaler.fit(trainX)

    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)
    return trainX, trainy, testX, testy

def dataset_shape():
    # load and summarize the dataset
    from collections import Counter
    # load the csv file as a data frame
    dataframe = read_csv(filename)
    dataframe.astype('int32').dtypes
    # summarize the shape of the dataset
    print(dataframe.shape)
    # summarize the class distribution
    target = dataframe.values[:,0]
    counter = Counter(target)
    for k,v in counter.items():
    	per = v / len(target) * 100
    	print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))

def train_model(X_train, X_test, y_train, y_test):
    # Determine Runtime for Models
    start = timeit.default_timer()
    # Define Models
    #classifier = KNeighborsClassifier(n_jobs=-1, n_neighbors=1)
    classifier = ExtraTreesClassifier(n_jobs=-1, n_estimators=1000)
    #classifier = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
    #classifier = BaggingClassifier(n_jobs=-1, n_estimators=1000)
    #classifier = SVC()
    # Fit Models
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    stop = timeit.default_timer()
    # Print Confusion Matrix and Classification Report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=5))
    print('Time: ', stop - start) 
    #fig, ax = plt.subplots(figsize=(10, 10))
    disp = plot_confusion_matrix(classifier, X_test, y_test, display_labels=[0,1,2,3], cmap=plt.cm.Blues, normalize=None)
    return classifier

def get_k(X_train, X_test, y_train, y_test):
    # Determine Optimal Value of K (1 <= K <= 40) for KNeighborsClassifier
    error = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))
    # Plot Error for Values of K
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

# Evaluate Model
def evaluate_model_cv(X, y, model):
    # Define KFold Cross-Validation Evaluation Metric
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # Evaluate Model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

# Test Models
def get_models_cv():
    models, names = list(), list()
    # KNN
    models.append(KNeighborsClassifier(n_jobs=-1, n_neighbors=1))
    names.append('KNN')
    # ET
    models.append(ExtraTreesClassifier(n_jobs=-1, n_estimators=1000))
    names.append('ET')
    # RF
    models.append(RandomForestClassifier(n_jobs=-1, n_estimators=1000))
    names.append('RF')
    # Bagging
    models.append(BaggingClassifier(n_jobs=-1, n_estimators=1000))
    names.append('BAG')
    # SVM
    models.append(SVC())
    names.append('SVM')
    return models, names

def train_cnn():
    start = timeit.default_timer()
    # LOAD DATA (80% training, 20% validation)
    raw_data = np.loadtxt(filename, skiprows = 1, dtype = 'int', delimiter = ',')
    x_train, x_val, y_train, y_val = train_test_split(raw_data[:,1:], raw_data[:,0], test_size = 0.2)
    # Reshape image data as 16x16 matrix
    x_train = x_train.reshape(-1, 16, 16, 1)
    x_val = x_val.reshape(-1, 16, 16, 1)
    # Sets pixel values range from (0-255) to (0-1) for faster performance
    # Set to 870 for activities, 891 for yoga 
    x_train = x_train.astype("float32") / 870.
    x_val = x_val.astype("float32") / 870.
    # One-hot encoding to convert labels of images for faster performance
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
        
    # TRAIN MODEL (Sequential API, Conv2D convolutional layers, MaxPooling layers, batch normalization)
    model = models.Sequential()

    model.add(layers.Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', input_shape = (16, 16, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPool2D(strides = (2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPool2D(strides=(2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1024, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    
    # Set first argument to 7 for activities, 8 for yoga
    model.add(layers.Dense(7, activation = 'softmax'))
    
    # Augmentation to improve generalization (i.e. generate additional training data by randomly perturbing images)
    datagen = preprocessing.image.ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)
    # Compile model (Loss function = categorical_crossentropy, Adam for fast optimization)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr = 1e-4), metrics = ["accuracy"])
    
    checkpoint = ModelCheckpoint("activities_model_weights.h5", monitor="accuracy", verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
    callbacks_list = [checkpoint, es]

    model.fit(x_train, y_train, epochs=200, verbose=1, callbacks=callbacks_list)
    # Train with lower learning rate for convergence. Increase learning rate, and then decrease learning rate by 10% per epoch
    annealer = callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    # Train on small validation set
    '''
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=20, #Increase this when not on Kaggle kernel
                           verbose=1,  #1 for ETA, 0 for silent
                           validation_data=(x_val[:400,:], y_val[:400,:]), #For speed
                           callbacks=[annealer])
    '''
    # EVALUATE MODEL 
    #Validate performance on entire 20% validation set
    final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
    print("Final loss: {0:.7f}, final accuracy: {1:.7f}".format(final_loss, final_acc))
    y_hat = model.predict(x_val)
    y_pred = np.argmax(y_hat, axis=1)
    y_true = np.argmax(y_val, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    # serialize model structure to JSON
    model_json = model.to_json()
    with open("activities_model.json", "w") as json_file:
        json_file.write(model_json)
    return model

def create_model():
    X_train, X_test, y_train, y_test = load_dataset(filename)
    return train_model(X_train, X_test, y_train, y_test)
    
def get_knn():
    X_train, X_test, y_train, y_test = load_dataset(filename)
    get_k(X_train, X_test, y_train, y_test)
    
def calculate_cv():
    # Load Dataset
    X, y = load_dataset_cv(filename)
    
    # Define Models
    models, names = get_models_cv()
    results = list()
    
    # Evaluate Models
    for i in range(len(models)):
        # Evaluate Performance of Models
        scores = evaluate_model_cv(X, y, models[i])
        results.append(scores)
        # Print Performance of Models
        print('>%s %.7f (%.7f)' % (names[i], mean(scores), std(scores)))
        
    # Plot Model Results
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()

def create_cnn():
    # Generate Training and Testing Data
    raw_data = np.loadtxt(filename, skiprows = 1, dtype = 'int', delimiter = ',')
    x_train, x_val, y_train, y_val = train_test_split(raw_data[:,1:], raw_data[:,0], test_size = 0.2)
    #trainX, trainy, testX, testy = load_dataset_cnn()
    # Reshape image data as 16x16 matrix
    x_train = x_train.reshape(-1, 16, 16, 1)
    x_val = x_val.reshape(-1, 16, 16, 1)
    # Sets pixel values range from (0-255) to (0-1) for faster convergence
    # Set to 870 for activities, 891 for yoga 
    x_train = x_train.astype("float32") / 870.
    x_val = x_val.astype("float32") / 870.
    # One-hot encoding to convert labels of images for faster performance
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    
    # Fit Model with Data
    model = train_cnn()
    rounded_y_val=np.argmax(y_val, axis=1)
    y_pred = model.predict(x_val, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(rounded_y_val, y_pred_bool, digits=5))
    return model

def predict_test():
    model = create_model()
    # Load Dataset
    test_dataset = pd.read_csv(test_filename)
    X_test = test_dataset.values
    #X_test, y_test = test_dataset[:,1:], test_dataset[:,0]
    # Scale Data
    scaler = StandardScaler()
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    
    y_pred = model.predict(X_test)
    np.savetxt("predictions_activities.csv", y_pred, delimiter=",")
    '''
    # Print Confusion Matrix and Classification Report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=5))
    #fig, ax = plt.subplots(figsize=(10, 10))
    disp = plot_confusion_matrix(model, X_test, y_test, display_labels=[0,1,2,3], cmap=plt.cm.Blues, normalize=None)
    return model
    '''

def predict_test_cnn():
    model = create_cnn()
    
    # Generate Training and Testing Data
    X_test = np.loadtxt(test_filename, skiprows = 1, dtype = 'int', delimiter = ',')
    # Reshape image data as 16x16 matrix
    X_test = X_test.reshape(-1, 16, 16, 1)
    
    y_pred = model.predict(X_test)
    y_pred_bool = np.argmax(y_pred, axis=1)
    np.savetxt("predictions_activities_cnn.csv", y_pred_bool, delimiter=",")
    

if __name__ == '__main__':
    ### Run this code to see the shape and features of the dataset
    #dataset_shape()
    
    ### Run this code to train model and obtain precision, recall, F1 score, and confusion matrix ###
    #model = create_model()
    
    ### Run this code to train CNN model ###
    #model = create_cnn()
    
    ### Run this code to train model to obtain CV accuracy ###
    #calculate_cv()
    
    ### Run this code to find optimal K value for KNN classifier ###
    #get_knn()
    
    ### Run this code to predict test data using KNN ###
    predict_test()
    
    ### Run this code to predict test data using CNN ###
    #predict_test_cnn()