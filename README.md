# Optimizing Trading Strategies using Deep Learning Classifiers

### Introduction
This project follows a structured Python project approach to explore and optimize trading strategies using historical stock data from the aapl_5m_train.csv and aapl_5m_validation.csv datasets. The aim is to compare the effectiveness of Technical Analysis, Machine Learning (ML), and Deep Learning (DL) models in devising profitable trading strategies.


The project adheres to a clear Python project structure and workflow. It involves:

#### 1.Data Preparation:
The datasets are loaded and divided into training and validation sets for model optimization and comparison.
#### 2.Model Development: 
Three Deep Learning Classifiers - MLP, RNN, and optionally CNN - are fitted as trading strategies using various sets of independent variables and target variables.
#### 3.Strategy Combination:
All possible combinations of these models are created, resulting in seven unique trading strategies for backtesting.
#### 4. Backtesting and Optimization: 
Strategies are backtested, tracking operations, cash, and portfolio value time series. Top-loss, take-profit, trade volume, and hyperparameters are fine-tuned to maximize profitability within defined bounds using the training dataset.
## Strategy Evaluation
#### °Optimal Strategy Selection: 
After optimization, the most profitable strategy is selected, comprehensively described in terms of employed models, trade signal generation, and other pertinent details.
## Validation and Comparison
#### °Validation Dataset Usage: 
The optimal strategy, without re-fitting the model, is applied to the validation dataset.
#### °Comparison to Passive Strategy: 
Results from the optimal strategy are compared against a passive strategy for performance evaluation.
## Project Documentation
#### °Jupyter Notebook Integration: 
All findings, including the list of operations, candlestick charts, employed models, trading signals, cash and portfolio value over time, and significant visualizations, are compiled into the same Jupyter notebook as the previous project.
#### °Conclusions and Insights:
The conclusions drawn from the results, insights into the effectiveness of different strategies, and observations regarding the performance of the optimal strategy versus the passive approach are detailed.
### Repository and Conclusion
GitHub Repository: The project's code, documentation, and findings are made available in a GitHub repository for reference and future analysis.
## Conclusion
This project rigorously explores the application of Deep Learning Classifiers in devising profitable trading strategies. It provides a detailed comparative analysis of strategies using training and validation datasets, culminating in the identification and validation of an optimal strategy for trading Apple stock.



