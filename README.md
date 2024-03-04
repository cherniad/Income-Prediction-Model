# Income Prediction Model

 __Introduction__ 

 The purpose of this project is, to build a model that would predict income and to show 
 what parameteres are the most influential. 

 
__Libraries used:__
- numpy
- matplotlib
- pandas
- seaborn
- scikit-learn

__Steps:__

1. Convert the csv file "income.csv" in the dataframe and explore the data

2. Process the data:
     a) drop the columns not needed for the model
     b) transform long data into wide data
     c) turn the values into binary features

3. Filter the data by dropping the columns with the weakest correlations

4. Create a heatmap and observe the data

5. Fit model in RandomForestClassifier
   
6. Use hyper parameter tuning to see which combination of parameteres gives the best performance

8. Create a visualisation with top-10 most important features in relation to income in matplotlib

9. Show partial dependence of the three most important features with sklearn.inspection



__Results and Learning Outcomes__ 

The model could be used for a market research or in HR departments to predict the income of potential employees. 
In the future, the function could be added to the code to allow for interactive input of values, 
further enhancing the usability of the model.
 
