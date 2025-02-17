import matplotlib.pyplot as plt
import streamlit as st

def plot_train_test_predictions(y_real, y_predict, date_test):
    '''
    Plots the actual and predicted values of actual and predicted sets in the same plot.

    Parameters:
    y_real (numpy.ndarray): Actual values of the data set.
    y_predict (numpy.ndarray): Predicted values of the data set.
    date_test (pandas.core.indexes.datetimes.DatetimeIndex): Dates of the data set.
    '''
    
    fig, ax = plt.subplots(figsize=(16, 8))  # Create the figure and axis explicitly
    ax.plot(date_test, y_real, color='m', label='Actual Test Data')
    ax.plot(date_test, y_predict, color='k', label='Predicted Test Data')
    ax.set_title('Actual vs Predicted Test Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature/Value')  # Adjust label as needed
    ax.legend()
    
    return fig  # Return the figure to be used with st.pyplot()

