# Rain Predictor 

In this project I will using the Australian rainfall data to practice on Machine Learning classification algorithms.
The models that will be employed are:

<ol>
  <li> KNN - K-Nearest Neighbourhood </li>
  <li> Decision Trees </li>
  <li> Logistic Regression</li>
  <li> SVM </li>
</ol>

## Table of contents
* [About the Dataset](#about-the-Dataset)
* [How to Run It](#how-to-run-it)
* [Results](#results)

# About the Dataset

The original source of the data is Australian Government's Bureau of Meteorology and the latest data can be gathered from [http://www.bom.gov.au/climate/dwo/](http://www.bom.gov.au/climate/dwo/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01).

The dataset to be used has extra columns like 'RainToday' and our target is 'RainTomorrow', which was gathered from the Rattle at [https://bitbucket.org/kayontoga/rattle/src/master/data/weatherAUS.RData](https://bitbucket.org/kayontoga/rattle/src/master/data/weatherAUS.RData?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01)

This dataset contains observations of weather metrics for each day from 2008 to 2017. The **weatherAUS.csv** dataset includes the following fields:

| Field         | Description                                           | Unit            | Type   |
| ------------- | ----------------------------------------------------- | --------------- | ------ |
| Date          | Date of the Observation in YYYY-MM-DD                 | Date            | object |
| Location      | Location of the Observation                           | Location        | object |
| MinTemp       | Minimum temperature                                   | Celsius         | float  |
| MaxTemp       | Maximum temperature                                   | Celsius         | float  |
| Rainfall      | Amount of rainfall                                    | Millimeters     | float  |
| Evaporation   | Amount of evaporation                                 | Millimeters     | float  |
| Sunshine      | Amount of bright sunshine                             | hours           | float  |
| WindGustDir   | Direction of the strongest gust                       | Compass Points  | object |
| WindGustSpeed | Speed of the strongest gust                           | Kilometers/Hour | object |
| WindDir9am    | Wind direction averaged of 10 minutes prior to 9am    | Compass Points  | object |
| WindDir3pm    | Wind direction averaged of 10 minutes prior to 3pm    | Compass Points  | object |
| WindSpeed9am  | Wind speed averaged of 10 minutes prior to 9am        | Kilometers/Hour | float  |
| WindSpeed3pm  | Wind speed averaged of 10 minutes prior to 3pm        | Kilometers/Hour | float  |
| Humidity9am   | Humidity at 9am                                       | Percent         | float  |
| Humidity3pm   | Humidity at 3pm                                       | Percent         | float  |
| Pressure9am   | Atmospheric pressure reduced to mean sea level at 9am | Hectopascal     | float  |
| Pressure3pm   | Atmospheric pressure reduced to mean sea level at 3pm | Hectopascal     | float  |
| Cloud9am      | Fraction of the sky obscured by cloud at 9am          | Eights          | float  |
| Cloud3pm      | Fraction of the sky obscured by cloud at 3pm          | Eights          | float  |
| Temp9am       | Temperature at 9am                                    | Celsius         | float  |
| Temp3pm       | Temperature at 3pm                                    | Celsius         | float  |
| RainToday     | If there was rain today                               | Yes/No          | object |
| RainTomorrow  | If there is rain tomorrow                             | Yes/No          | float  |

Column definitions were gathered from [http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml](http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01)


# How to Run It

To run this project you can simply clone clone it and run it with an interpreter (e.g. Spyder).

# Results
The results of the project are presented through the confusion matrices of the four different models.

![image](https://github.com/user-attachments/assets/0cdeb48f-779d-4e48-b8a2-32005da6e22b)
![image](https://github.com/user-attachments/assets/c2026a67-7258-4160-bfed-8b17c519bfa3)  
![image](https://github.com/user-attachments/assets/235132bd-4d5e-4faa-82c3-7bf2334bd1fb)
![image](https://github.com/user-attachments/assets/062d3e64-cc1f-4a11-b25f-e76b0d11d918)

Overall, while all four models demonstrate reasonable predictive capabilities, the Logistic Regression (LR) model stands out as the most effective in accurately forecasting the likelihood of rain. It consistently delivers the best balance between true positives and true negatives, indicating strong performance in both predicting rainy and non-rainy days.    
  
On the other hand, the fourth model, even after extensive tuning, has not yielded satisfactory results. Alarmingly, this model seems to fail at making varied predictions, as it consistently outputs the same prediction for every instance, effectively rendering it unusable for accurate weather forecasting. This suggests a critical issue, potentially with the model's implementation or underlying logic, that needs to be addressed.



