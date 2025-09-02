# Neural Network Interpretability for Titanic Survival

This project investigates the "black box" nature of a neural network by comparing its predictions on the Titanic dataset to a simple, transparent Logistic Regression model.
It uses a **Local Linear Approximation** technique to explain *why* the neural network makes a specific prediction for an individual passenger.

## Project Goal 

The main goal is to show that while a neural network is more accurate, we don't have to treat it as a complete black box.
By approximating its local behavior, we can gain insights similar to those from simpler models, helping to bridge the gap between performance and interpretability.

## Key Findings 

* **Context is Key**: The neural network can change the importance of features based on a passenger's specific data.
  For example, it correctly predicted a female passenger in 3rd class would die, weighting `pclass` more heavily than `sex`.
  The Logistic Regression model incorrectly predicted she would survive based on its global rule that "females are more likely to survive".
* **Local vs. Global**: This project demonstrates the difference between the neural network's flexible, **local** explanations for each passenger and the logistic regression's rigid, **global** explanation that applies to everyone.

## How It Works

1.  A **Neural Network** and a **Logistic Regression** model are trained to predict Titanic survival.
2.  For any given passenger, the path of active neurons in the neural network is used to create a temporary linear model that explains that single prediction.
3.  This "local explanation" is visualized and compared to the global coefficients of the Logistic Regression model.

For a full breakdown of the methodology, performance metrics, and case studies, please see the complete Interpretability Investigation Report in this repository.
