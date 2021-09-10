# Using Machine Learning to Determine Land Cover from Satellite Imagery

This project was completed as part of a bachelors thesis project (Samuel Jordan, University of Southampton, 2021).

This readme gives a brief rundown of the project pipeline and results.

&nbsp;
## Satellite Imagery Source Data

The satellite imagery used was an ESA Sentinel level-2A product. Level-2A provides bottom of atmosphere reflectance images, meaning the image has already been pre-processed to remove any distortions caused by the atmosphere. This image includes 13 different spectral bands, although only four bands have 10m resolution. These bands include the visible spectrum as well as near infrared (NIR). Only 10m resolution bands were used throughout the project to avoid having to upscale any images and to maximise the accuracy of the satellite imagery reflectance data. Sentinel data products are made available systematically and free of charge to all data users including the general public, scientific and commercial users.

The area of interest (AOI) is a section of the Isle of Wight and some surrounding ocean, about 191km<sup>2</sup>. It was captured on the 30<sup>th</sup> of July 2020. The full image and the AOI can be seen in figures 1 & 2. This AOI was selected for its minimal cloud cover and balance of land cover types, including ocean, beach, urban space, fields and vegetated space. The image is 1804x1058 or 1908632 pixels.

Each band is saved separately in a jpeg200 format with 16-bit colour depth. Using QGIS, a geographic information system application, the bands were combined into one image and the image fully georeferenced. The means a coordinate system is assigned to the image, so that it can easily be aligned with the ground truth data. The file is saved as a geotiff.

&nbsp;
[full](images/full.png)
*Figure 1 - The full sentinel 2 datastrip. AOI outlined in red.*

&nbsp;
[AOI](images/AOI.png)
*Figure 2 - The area of interest*

&nbsp;
## Ground Truth

The ground truth data is created using QGIS. The data contains vector points, each attributed a coordinate and one of five different classes shown in Table 1. The truth data is overlayed Infront of the AOI in figure 3. While five classes certainly do not offer detailed information with regards to the land cover, it is enough to accurately classify all the land in the AOI and is adequate for the scope of this project. A basic set of classes means points are unlikely to be mislabelled due to ambiguity in the image or a lack of specialist knowledge if different types of forest, grassland or farmland were to be classified, for example. Furthermore, all these classes are discernible at the 10m resolution offered by sentinel 2 imagery.

&nbsp;
[truth](images/truth.png)
*Figure 3 - Truth labels overlayed on the AOI.*

&nbsp;
| Class       | Description                                                        | Frequency   | Training    | Testing     |
| ----------- | ------------------------------------------------------------------ | ----------- | ----------- | ----------- |
| Sand        | Any sandy area, mainly beaches.                                    | 55          | 45          | 10          |
| Urban       | Residential/commercial/industrial areas, roads.                    | 156         | 103         | 53          |
| Forest      | Any kind of dense vegetation, mainly broadleaf forests.            | 121         | 86          | 35          |
| Fields      | Grassland, meadows, farmland, pastures, heathland.                 | 432         | 304         | 128         |
| Water       | Any body of water. Oceans, estuaries and inland water.             | 213         | 146         | 67          |

*Table 1 – A list of classes used in the ground truth data, their descriptions and frequency.*

&nbsp;
## Feature Engineering

In a pixel-based approach, the model is trained and makes predictions based upon the four reflectance values per pixel. This means the model can only look at one pixel at a time, with no concern for other pixels in the surrounding area. An object-based approach allows the model to utilise some spatial data when training and classifying. A felzenswalb segmentation algorithm is used to generate segments of the satellite image. It was implemented using the Scikit Image python library. The mean reflectance value is calculated for each segment. Figure 4 shows the result of the segmentation. This is the data that will be used to train the model.

&nbsp;
[segments](images/segments.png)
*Figure 4 - The result of the segmentation. Created in QGIS using the palleted/unique value render type with random colours.*

&nbsp;
## Model Training

For the purposes of training and validation, only segments with truth data can be used. That is, using the ground truth data, any segments with a point within them are assigned the relevant class.

To train the model, the data must be split into a training and a testing (or validation) dataset. The data is split such that the truth and training data set share similar proportions of classes. It is otherwise split randomly, and not by location, so as to avoid biasing the training/testing data towards one part of the image. The classified segments are assigned to the training and test datasets by a ratio of 7:3 respectively. The use of a training/testing split is vital. By optimising the model to maximise accuracy on data it trained from, the model would overfit and as a result underperform when introduced to new data. As such, validation of the model must use data that has not been seen during training.

A grid search method is used to optimise the hyperparameters of the algorithms. This involves an exhaustive search across a subset of the hyperparameter space in order to find the best set of hyperparameters. A subset of kernels and C values were searched before settling upon an RBF kernel and C = 100.

&nbsp;
## Results

The final model predicts with 0.959% agreement and a kappa score of 0.943 upon the test data. Such results are likely in part to this being a simple 5-class classification problem of mostly distinct classes but is nevertheless a good result. See figure 5 for the classified AOI.

&nbsp;
[classified](images/classified.png)
*Figure 5 - The result of the classification. Created in QGIS using the palleted/unique value render type and a custom style.*
