# Motion detection using sensor data

# Tasks:
-The stated task is "Use ML model to determine when a Person is near to IoT Device based on reading recorded on device's sensors.

-As we don't have labels of time stamps where people are near the IoT device we are limited to unsupervised methods.

-We do however, have three different devices in different locations, each with about a week of time series data recorded at a sampling interval of 5-10 seconds between               measurements.

# The proximity of a person could affect the recorded parameters in the various ways:

-CO levels be impacted by the presence of an individual

-Humidity might increase if a person is exhaling into the room

-Light levels might drop if a person is occluding the light sensor. Light levels might increase if a person turns on a light or opens a closet door

-LPG levels might drop if a person breathes it in and thereby filters it from the local atmosphere

-The movement of an individual near to a sensor would create detectable motion, though some information about other sources of motion/vibration would be needed to attribute it to a nearby person

-Again, smoke levels could be reduced by the filtering effect of a persons lungs or increased if a person lights up a cigarette infront of the smoke sensor

-The ambient temperature might be increased by the presence of a person(s) next to the temperature sensor for a period of time.

# Approach:

-Lets plot the time series with a meaningful time scale (time of day might indicate when people are more likely to be nearby)

-Lets find out if there are any daily patterns the data

-Lets see if there are any significant differences between the sensor time series between the three locations

-Attempt to define thresholds/confidence intervals/clustering on windows of time series data to define plausible "humans are nearby" intervals (i.e corresponding to light/motion/atmospheric/temperature changes)

![image](https://user-images.githubusercontent.com/58647922/120797595-e0d8d500-c559-11eb-8e99-caf3af7e9a31.png)
![image](https://user-images.githubusercontent.com/58647922/120797718-05cd4800-c55a-11eb-9171-ea9c1cd80ff1.png)
![image](https://user-images.githubusercontent.com/58647922/120797749-11207380-c55a-11eb-8fe9-a39532a48b41.png)
![image](https://user-images.githubusercontent.com/58647922/120797801-20072600-c55a-11eb-8fb2-2d340a57d818.png)
![image](https://user-images.githubusercontent.com/58647922/120798011-5e044a00-c55a-11eb-9eaf-95f27076be6a.png)

# From the plot above we can see a few things:

-CO, LPG and smoke levels (air quality metrics) are correlated for each device and vary over the time series and between devices.

-Dramatic swings in temperature are recorded (are they real or the result of sensor malfunction?) as well as more moderate temperature oscillations.

-There are spikes in humidity and motion.

-Illumination is either continuous or transient

![image](https://user-images.githubusercontent.com/58647922/120798378-e84cae00-c55a-11eb-92b6-3b4ab6f18449.png)

From the boxplots above we can see that the three devices are definitely located in distinct environments as:

-Ambient air pollution levels are highest in b8, followed by 1c and 00. 00 is less polluted most of the time but has more significant spikes of air pollution than the other devices.

-The three devices are in locations will slightly different average temperatures, in the range of 20-30 degC. 00 and 1c show significant temperature drop outliers.

-The three devices are also in locations with different humidity levels, in the range of roughly 50-75%. All three have some outliers showing increases and decreases in humidity which for 1c and 00 are substantial (65 to 0%)

# Any daily patterns in the data?

-We can use facebook prophet to easily (if not rapidly) calculate and plot the hourly trends for our data. Lets take smoke across the three devices as an example

![image](https://user-images.githubusercontent.com/58647922/120799030-d15a8b80-c55b-11eb-8f93-df42aaf1b216.png)

![image](https://user-images.githubusercontent.com/58647922/120799212-1252a000-c55c-11eb-95bf-17c0a7a80891.png)



-From the fbprophet model above we can see that there is a trend for a fall in smoke levels from around 6am to 8 pm and an increase in smoke levels around midnight each day, with the same trend seen across the three device locations

-We could go through each sensor type and generate trend data as above. You could also groupby using the datetime column and group over a day time frame. From this you could calculate the mean and calculate confidence intervals, giving you similar trend information to that generate by fbprophet

-These confidence intervals could be the basis for an anomaly detection system (smoke alarm, human alarm etc)

# Unsupervised learning to identify time series windows where humans are near

The central question here is defining which aspects of the time series can be attributed to human activity? It could be defined by:

-motion spikes - when humans are near they wobble the accelerometer

-light spikes - when humans open a door light falls on the detector

-spikes in air pollution - when humans drive up to a sensor or turn on a machine fumes are produced

-temperature and humidity spikes - when humans open a door the temperature and humidity shift accordingly

# Pre-Processing and Dimensionality Reduction

-Scale the data

-Perform PCA and look at the most important principal components based on inertia

![image](https://user-images.githubusercontent.com/58647922/120801068-53e44a80-c55e-11eb-92bf-6a1a7356f706.png)

-It appears that the first two principal components are the most important as per the features extracted by the PCA in above importance plot. So as the next step, I will perform PCA with 2 components which will be my features to be used in the training of the models.

-Running the Dickey Fuller test on the 1st principal component, I got a p-value of 2.3283893641009937e-16 which is very small number (much smaller than 0.05). Thus, I will reject the Null Hypothesis and say the data is stationary. I performed the same on the 2nd component and got a similar result. So both of the principal components are stationary which is what I wanted.

Let’s start training with these algorithms.

# Interquartile Range

-Calculate IQR which is the difference between 75th (Q3)and 25th (Q1) percentiles.

  outlier_lower = Q1 - (1.5*IQR)
  
  outlier_upper = Q3 + (1.5*IQR)

-Calculate upper and lower bounds for the outlier.

-Filter the data points that fall outside the upper and lower bounds and flag them as outliers.

-So, We are getting 510 outliers based on pc1 and 425 based on pc2 which we will consider as anomaly.

# K-Means Clustering

-Calculate the distance between each point and its nearest centroid. The biggest distances are considered as anomaly.

![image](https://user-images.githubusercontent.com/58647922/121302234-03327000-c917-11eb-8f06-bc4e96c7a793.png)

-We use outliers_fraction to provide information to the algorithm about the proportion of the outliers present in our data set. Situations may vary from data set to data set. However, as a starting figure, I estimate outliers_fraction=0.02 (2% of df are outliers as depicted).

-Calculate number_of_outliers using outliers_fraction.

-Set threshold as the minimum distance of these outliers.

-The anomaly result of anomaly1 contains the above method Cluster (0:normal, 1:anomaly).

-Visualize anomalies with Time Series view.

using pc1 and pc2 as 2 features:

![image](https://user-images.githubusercontent.com/58647922/121302558-818f1200-c917-11eb-9fb7-6b1ca621ee48.png)

using co and humidity as 2 features as (temperature and humidity) as well as (co, lpg, and smoke are highly correlated):

![image](https://user-images.githubusercontent.com/58647922/121302904-0417d180-c918-11eb-8821-c605876ccdcc.png)

Anomalies in time series data based on K-Means plot:

![image](https://user-images.githubusercontent.com/58647922/121303040-3295ac80-c918-11eb-8d39-48bd966b3c04.png)

![image](https://user-images.githubusercontent.com/58647922/121303098-48a36d00-c918-11eb-8e02-4e444e814191.png)

![image](https://user-images.githubusercontent.com/58647922/121303159-5f49c400-c918-11eb-8cd7-e88ee41b0eb3.png)

![image](https://user-images.githubusercontent.com/58647922/121303171-65d83b80-c918-11eb-81c1-bdaaeda9087d.png)

![image](https://user-images.githubusercontent.com/58647922/121303185-6bce1c80-c918-11eb-8897-73faaeb0b64d.png)

# Isolation Forest:

![image](https://user-images.githubusercontent.com/58647922/121304771-6245b400-c91a-11eb-81a2-164ca7d126d8.png)

# Model Evaluation:

-It is interesting to see that all three models detected a lot of similar anomalies.

-IQR detected far less anomalies than K-Means and Isolation forest. Hence we will take outliers detected in the latter 2 models as our anomalies. 

-Just by visually looking at the above graphs, one could easily conclude that the Isolation Forest might be detecting a more anomalies than the K-Means.

-But More than 85% of points detected by later 2 models are similar.

# Conclusion:

-Finally, We write a function which takes device number and cleaned data as parameters and would return dataframe containing anomalies detected by K-Means and Isolation forest model along with time series curve containing anomalies by both models.
  
-Running it for device 3 returns following points as anomaly:
  
  ![image](https://user-images.githubusercontent.com/58647922/121306909-0af51300-c91d-11eb-92e2-534a0715794c.png)
  
  ![image](https://user-images.githubusercontent.com/58647922/121307471-a7b7b080-c91d-11eb-8b27-bd6faf48e01f.png)

  
  ![image](https://user-images.githubusercontent.com/58647922/121306953-15afa800-c91d-11eb-8dd8-dc99aeb5a312.png)
  
  ![image](https://user-images.githubusercontent.com/58647922/121307513-b0a88200-c91d-11eb-8b68-ac41c6d247ce.png)
  
  ![screenshot-github com-2021 06 09-12_18_51](https://user-images.githubusercontent.com/58647922/121307546-b900bd00-c91d-11eb-8382-9ed5cc36bb40.png)


