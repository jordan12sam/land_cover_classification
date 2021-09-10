import os
import time

import numpy as np
import scipy as sp
import geopandas as gpd
import pandas as pd
import pickle
from osgeo import gdal, ogr
from skimage import exposure
from skimage.segmentation import felzenszwalb
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt


#given its pixel data, return the pixel statistics of a segment
def segment_stats(segment_pixels):

    features = []
    npixels, nbands = segment_pixels.shape

    #calculate stats for each band
    for b in range(nbands):
        #calculate a list of pixel stats for the given band
        #includes size, minmax, mean, variance, skew, kurtosis
        stats = sp.stats.describe(segment_pixels[:, b])

        #only stats[2] is used (the mean)
        #could be extended to use others too
        band_stats = [stats[2]]

        #append to the list of stats for the segment
        features += band_stats

    return features

def open_AOI():
    #open image of the 'Area of Interest'
    #all bands should be in the same resuolution
    fp = 'data/AOI.tif'
    ds = gdal.Open(fp)
    nbands = ds.RasterCount
    band_data = []
    print('bands', ds.RasterCount, 'rows', ds.RasterYSize, 'columns', ds.RasterXSize)

    #arrange raster data into a numpy array
    for i in range(1, nbands+1):
        band = ds.GetRasterBand(i).ReadAsArray()
        band_data.append(band)
    img = np.dstack(b for b in band_data)
    print(img.shape)
    return img, ds

def feature_extraction(img, ds):

    #segment the image using a felzenswalb segmentation
    #experiment with parameters of felzenswalb() to find an adequate segmentation
    seg_start = time.time()
    segments = felzenszwalb(img, scale=3, sigma=0.0, min_size=5)
    print(f"number of segments: {len(np.unique(segments))}")
    print('segments complete', time.time() - seg_start)

    #save segments as an image
    segments_fp = 'data/segments.tif'
    driverTiff = gdal.GetDriverByName('GTiff')
    segments_ds = driverTiff.Create(segments_fp, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    segments_ds.SetGeoTransform(ds.GetGeoTransform())
    ds.SetProjection(ds.GetProjectionRef())
    segments_ds.GetRasterBand(1).WriteArray(segments)
    segments_ds = None

    #create a list of unique segment ids
    obj_start = time.time()
    segment_ids = np.unique(segments)

    #create lists to store segment stats and ids respectively
    objects = []
    object_ids = []

    #loop through list of unique segment ids
    for id in segment_ids:

        #filter for pixels with a matching segment id
        segment_pixels = img[segments == id]

        #calculate the pixel stats for the segment
        #append to the list of stats 'objects'
        object_feature = segment_stats(segment_pixels)
        objects.append(object_feature)
        object_ids.append(id)

        #print progress every 2500 segments
        if id%2500 == 0:
            print('pixels for id', id)

    print('created', len(objects), 'objects with', len(objects[0]), 'variables in', time.time()-obj_start, 'seconds')

    #save objects
    with open('data/objects.txt', 'wb') as object_path:
        pickle.dump(objects, object_path)
    with open('data/object_ids.txt', 'wb') as object_path:
        pickle.dump(object_ids, object_path)

    return segments, objects

def train_test_split():
    #load truth data
    #stored as a set of labels containing land cover type and geo-coordinate
    gdf = gpd.read_file('data/truth/truth.shp')

    #assign each unique land cover type an id
    class_names = gdf['land_cover'].unique()
    print('class names:', class_names)
    class_ids = np.arange(class_names.size) + 1
    gdf['id'] = gdf['land_cover'].map(dict(zip(class_names, class_ids)))

    #seperate truth data into training and testing sets
    gdf_train = gdf.sample(frac=0.7)
    gdf_test = gdf.drop(gdf_train.index)
    print('gdf shape:', gdf.shape)
    print('training shape:', gdf_train.shape)
    print('testingshape:', gdf_test.shape)

    #save the train/test datasets
    train_fp = 'data/train/train.shp'
    test_fp = 'data/test/test.shp'
    for fn in [os.path.dirname(x) for x in [train_fp, test_fp]]:
        if not os.path.exists(fn):
            os.makedirs(fn)
    gdf_train.to_file(train_fp)
    gdf_test.to_file(test_fp)

    return train_fp, test_fp

def rasterise_data(fp, ds):
    #rasterise data
    #convert class labels into a raster of class ids
    #the pixel the label coincides with takes its class id
    #all over pixels take a 0 (no data)
    train_ds = ogr.Open(fp)
    lyr = train_ds.GetLayer()
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', ds.RasterXSize, ds.RasterYSize, 4, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(ds.GetGeoTransform())
    target_ds.SetProjection(ds.GetProjection())
    options = ['ATTRIBUTE=id']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)
    return target_ds.GetRasterBand(1).ReadAsArray()

def label_data(segments, objects, ground_truth):

    #get a list of class ids (except no data)
    classes = np.unique(ground_truth)[1:]

    segments_per_class = {}
    #for each class id, create a list of segment ids that share the class id
    for klass in classes:
        segments_of_class = segments[ground_truth == klass]
        segments_per_class[klass] = set(segments_of_class)


    #create lists to hold features and labels
    training_objects = []
    training_labels = []
    testing_objects = []
    testing_labels = []

    #for each class id
    for klass in classes:

        #get a list of objects belonging to the class id
        segment_ids = np.unique(segments)
        class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]

        #append the objects and their respective label to a pair of lists
        training_labels += [klass] * len(class_train_object)
        training_objects += class_train_object

    return training_objects, training_labels

def fit_model(training_objects, training_labels):

    #define support vector machine parameters
    classifier = SVC(C=10.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0)
    #train the svm using the training data
    classifier.fit(training_objects, training_labels)

    return classifier

def classify_AOI(classifier, segments, objects, ds):

    print('Predicting Classifications...')
    classifier_start = time.time()

    #predict the object classes
    predicted = classifier.predict(objects)
    print('Classified in ', time.time()-classifier_start)

    #map the class ids on to their respective segment ids
    print("Saving classified image...")
    segment_ids = np.unique(segments)
    clf = np.copy(segments)
    for segment_id, klass in zip(segment_ids, predicted):
        clf[clf == segment_id] = klass

    #save the classified image
    pred_fn = 'data/classified.tif'
    driverTiff = gdal.GetDriverByName('GTiff')
    clfds = driverTiff.Create(pred_fn, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    clfds.SetGeoTransform(ds.GetGeoTransform())
    clfds.SetProjection(ds.GetProjection())
    clfds.GetRasterBand(1).SetNoDataValue(0)
    clfds.GetRasterBand(1).WriteArray(clf)
    clfds = None

    print('Done.')

def accuracy(truth):
    #open predictions
    pred_ds = gdal.Open("data/classified.tif")
    pred = pred_ds.GetRasterBand(1).ReadAsArray()

    #get the indecies of pixels with data points
    idx = np.nonzero(truth)

    #create a confusion matrix comparing the true and predicted values
    cm = metrics.confusion_matrix(truth[idx], pred[idx])
    
    print('Matrix:\n', cm)

    #print performance metrics
    accuracy = (cm.diagonal() / cm.sum(axis=0))*100
    print('Prediction Accuracy:\n', accuracy)

    accuracy2 = (cm.diagonal().sum() / cm.sum(axis=0).sum())
    print('Total Accuracy:', accuracy2)

    kappa = metrics.cohen_kappa_score(truth[idx], pred[idx])
    print('Kappa Score:', kappa)

def main():
    img, ds = open_AOI()
    segments, objects = feature_extraction(img, ds)
    train_fp, test_fp = train_test_split()
    training_truth = rasterise_data(train_fp, ds)
    testing_truth = rasterise_data(test_fp, ds)
    train_x, train_y = label_data(segments, objects, training_truth)
    model = fit_model(train_x, train_y)
    classify_AOI(model, segments, objects, ds)
    accuracy(testing_truth)

if __name__ == "__main__":
    main()