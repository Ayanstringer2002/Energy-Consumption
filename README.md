# Energy Consumption Prediction

## Overview
This project focuses on predicting energy consumption using various machine learning models. The dataset used for this study is obtained from Fingrid's open data portal.

## Dataset
- **Source:** [Fingrid Open Data](https://data.fingrid.fi/en/data?datasets=364)
- **Description:** The dataset contains historical energy consumption records.

## Models Used
The following machine learning models were implemented for energy consumption prediction:
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Recurrent Neural Network (RNN)**
- **Long Short-Term Memory (LSTM)**
- **Random Forest**

## Evaluation Metrics
The performance of the models was evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**

## Model Performance
The results obtained for each model are as follows:

| Model          | MAE          | RMSE         |
|---------------|-------------|-------------|
| SVM          | 1336.963422  | 1659.793450 |
| Random Forest |  955.590299  | 1224.964659 |
| RNN          |  849.721955  | 1068.059504 |
| LSTM         |  965.219684  | 1256.041496 |

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Ayanstringer2002/Energy-Consumption
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```bash
   python ModelTraining.ipynb
   ```

## Conclusion
The LSTM model achieved the best performance in terms of both MAE and RMSE, making it the most suitable model for energy consumption prediction in this study.


