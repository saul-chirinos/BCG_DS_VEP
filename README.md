# European Energy Market Customer Churn Prediction

## Project Overview

### Business Problem

This project aims to help PowerCo, a fictional company, retain their customers. PowerCo is a major gas and electricity utility that supplies to corporate, SME (Small & Medium Enterprise), and residential customers. A fair hypothesis is that price changes affect customer churn. Therefore, it is helpful to know the probability a customer is to churn at their current price, for which a good predictive model could be useful. For the customers at risk of churning, a discount of 20% will be offered to help retain them.

### Approach

Design a machine learning pipeline where categorical features are encoded, numerical features are scaled, and the data is trained on predicting client churn, where 0 means they stayed and 1 means they churned. The three models experimented with are Logistic Regression, KNN, and Random Forest. From those models, Random Forest performed the best when comparing recall scores. Why recall? Recall measures a model's ability to collect all relevant documents.

$$
Recall = \frac{TP}{(TP+FN)}
$$

The goal is to decrease the frequency of $FN$, false negatives, which in this case are events where the model predicted the client did not churn when they did.

### Web API

The Random Forest model was integrated with a web API via *Streamlit* where hypothetical employees of PowerCo can utilize it to make predictions on client turnover. The API can be found [here](https://saul-chirinos-bcg-ds-vep-app-zou3vh.streamlit.app/).

![alt text](/Prototype/app_image.png)
