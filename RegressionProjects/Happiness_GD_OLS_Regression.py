# Author: Mohammad Nasser
# Created: 17th of Sept 2024
# Submission: Assigment 1 Part I


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

# linear regression using gradient descent
class GDLinearRegression:
    #initialize with learning rate and number of iterations
    def __init__(self, learning_rate=0.01, iterations=1000):
            
    # edge cases being tested
        if not (0<learning_rate<=1):
            raise ValueError("Learningg Rate must be between 0 and 1.")
        if iterations<=0:
            raise ValueError("Iterations must be positive.")
        
        np.random.seed(42) 
        self.learning_rate=learning_rate  # step size to learn for the GD
        self.iterations=iterations  # num of iteration to run GD
        self.beta_prime=None  # weights to be learned by model

    #fit model to data using gradient descent
    def fit(self, X, y):
    # edge cases being tested
        if len(X)!=len(y):
            raise ValueError("There is a mismatch in the number of samples between X and y.")
        
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("X or y contains NaN values.")
        
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("X or y contains infinite values.")
                             
        X_b=np.c_[np.ones((X.shape[0], 1)), X]  #add bias term

        self.beta_prime= np.random.rand(X_b.shape[1])  #initialize weights randomly
        
        #loop through the number of iterations
        for _ in range(self.iterations):

            #compute gradient using the formula from lec thats been derived
            gradients = 2/X_b.shape[0]* X_b.T.dot(X_b.dot(self.beta_prime)-y)
            self.beta_prime-= self.learning_rate* gradients  #update the weights 

        return self  

    #predict based on new data
    def predict(self, X):
        #add bias for prediction
        X_b=np.c_[np.ones((X.shape[0],1)),X]  
        return X_b.dot(self.beta_prime)  # return predicted values

# ols regression using the  closed-form solution
class OLSLinearRegression:
    # fit model using the OLS formula
    def fit(self, X, y):
        #add intercept term to X
        X_b=np.c_[np.ones((X.shape[0], 1)),X]  

        #calculate beta_prime using (X^T * X)^-1 *X^T*y
        self.beta_prime=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    #predict function to return predictions on new data
    def predict(self, X):
        # add intercept to input features
        X_b=np.c_[np.ones((X.shape[0], 1)),X] 
        return X_b.dot(self.beta_prime)  # return predicted values

def calculate_mse(y_true, y_pred):
    # calc MSE btwn true and predicted values
    mse= np.mean((y_true - y_pred) ** 2)  
    return mse

def select_best_gd_result(gd_results, y_true, metric="MSE"):
    #initializing the variables to track the error based on calculate_errors func
    best_result=None
    best_error= float('inf')  ## initial error is set to infinity 

    #iterate through GD results to evaluate the error
    for result in gd_results:
        # calc mse using the func
        error=calculate_mse(y_true, result['predictions'])
        
        # updating the model w a lower error
        if error<best_error:
            best_error=error
            best_result=result

    # Print the best model's error for the selected metric 
    print(f"Best GD model selected by {metric} with error: {best_error}")
    
    # return model with the lowest error based on the metric chosen 
    return best_result

def plot_gd_lines(X_normalized, y, gd_results):
    # white grid style for the plot
    sns.set(style="whitegrid")

    #scatter-plot of the data points
    plt.scatter(X_normalized, y, color='blue', label='Data points', s=40)

    #  unique colors and line styles to diff between the lines
    colors = sns.color_palette("husl", len(gd_results))  # auto assigns the colors 
    line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), '-']  # line styles

    # plot  the 6 of the GD results with diff parameters 
    for i, result in enumerate(gd_results[:6]):
        sns.lineplot(x=X_normalized.flatten(), y=result['predictions'], 
                     label=f"Learning Rate ={result['lr']}, Epochs={result['epochs']}",
                     linestyle=line_styles[i % len(line_styles)],  
                     # goes through each line styles and color as well as set a predefined width
                     color=colors[i % len(colors)],  
                     linewidth=2)

    #title and labels for the plot 
    plt.title("Gradient Descent Results with Different Learning Rates and Epochs", fontsize=14)
    plt.xlabel('Normalized GDP per capita', fontsize=12)
    plt.ylabel('Happiness Score', fontsize=12)
    
    # displays the legend and  the plot
    plt.legend(loc='best', fontsize=10)
    plt.show()

def plot_comparison(X_normalized, y, best_gd_result, ols_predictions):
    # seaborn white grid style for the plot
    sns.set(style="whitegrid")

    #scatter-plot of the data points
    plt.scatter(X_normalized, y, color='blue', label='Data points', s=40)

    # GD plot w red dashed line and markers
    sns.lineplot(x=X_normalized.flatten(), y=best_gd_result['predictions'], 
                 color='red', label=f" The Best GD (Learning Rate={best_gd_result['lr']}, Epochs={best_gd_result['epochs']})", 
                linestyle='--', linewidth=2, marker='o', markersize=7)

    # OLS plot w green and markets to diff
    sns.lineplot(x=X_normalized.flatten(), y=ols_predictions, 
                 color='green', label='OLS Line', 
                 linestyle='-', linewidth=2, marker='x', markersize=7)

  #title and labels for the plot
    plt.title("Comparison of Best Gradient Descent and OLS", fontsize=14)
    plt.xlabel('Normalized GDP per capita', fontsize=12)
    plt.ylabel('Happiness Score', fontsize=12)

    # displays the legend and the plot
    plt.legend(loc='best', fontsize=10)
    plt.show()

def load_and_preprocess_data(filepath):
    try:
        data=pd.read_csv(filepath)

        # filter for  2018 and remove rows with missing values in key columns
        filtered_data= data[(data['Year'] == 2018) & 
                            (~data['Cantril ladder score'].isna()) & 
                             (~data['GDP per capita, PPP (constant 2017 international $)'].isna())]

        # only happiness scores > 4.5
        filtered_data =  filtered_data[filtered_data['Cantril ladder score'] >4.5]

        #filter the data
        filtered_data= filtered_data[['Cantril ladder score','GDP per capita, PPP (constant 2017 international $)']]
        filtered_data.columns = ['Happiness', 'GDP']

        # feature matrix X & target var y
        X = filtered_data['GDP'].values.reshape(-1, 1)
        y = filtered_data['Happiness'].values

        #normalzing both GDP and happiness for the GD
        X_normalized = (X-np.mean(X))/np.std(X)
        y_normalized = (y-np.mean(y))/np.std(y)

        return X_normalized,y_normalized
    
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None,None
    except Exception as e:
        print(f"Error occurred during data loading and preprocessing: {e}")
        return None,None
    
def main():
    #load and pre process the data using helper func
    file_path= 'gdp-vs-happiness.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    
    X_normalized, y_normalized= load_and_preprocess_data(file_path)
    
    # make sure data is loaded and pre processed
    if X_normalized is None or  y_normalized  is None:
        return

    # train OLS model
    model_ols = OLSLinearRegression()
    model_ols.fit(X_normalized, y_normalized)
    ols_predictions = model_ols.predict(X_normalized)

    # test cases w diff parameters
    test_cases = [
        {"learning_rate":0.00001, "epochs":500},
        {"learning_rate":0.0001, "epochs":1000},
        {"learning_rate":0.67, "epochs":2000},  # Slower learning rate
        {"learning_rate":5e-6, "epochs":4000},  # Very small learning rate
        {"learning_rate":0.5, "epochs":2000},   # Large learning rate
        {"learning_rate":1, "epochs":1000}      # Aggressive learning rate
    ]

    # store the results for the GD
    gd_results = []

    # train the model based on the test cases
    for case in test_cases:
        lr= case["learning_rate"]
        epochs= case["epochs"]
        model_gd= GDLinearRegression(learning_rate=lr, iterations=epochs)
        model_gd.fit(X_normalized, y_normalized)  # train data

        #get the prediction
        predictions = model_gd.predict(X_normalized)  

        # store the results
        gd_results.append({
            'lr': lr,
            'epochs': epochs,
            'beta_prime': model_gd.beta_prime,
            'predictions': predictions
        })

    # plots the results for the GD
    plot_gd_lines(X_normalized, y_normalized, gd_results)

    #select the  best GD result based on MSE 
    best_gd_result=select_best_gd_result(gd_results, y_normalized, metric="MSE") 

    # comapre GD w OLS and plot it 
    plot_comparison(X_normalized, y_normalized, best_gd_result, ols_predictions)

    # Print beta' values for all GD
    print("Gradient Descent Beta Prime values:")
    for result in gd_results:
        print(f"Learning Rate ={result['lr']} | Epochs ={result['epochs']} | Beta Prime ={result['beta_prime']}")

    # print beta' values for the OLS and the best GD models
    print("OLS Beta Prime values:", model_ols.beta_prime)
    print(f"Best GD Beta Prime values: {best_gd_result['beta_prime']} with learning rate {best_gd_result['lr']} and epochs {best_gd_result['epochs']}")

if __name__ == '__main__':
    main()
