import os
import streamlit as st
import pandas as pd
from RF.data_preprocessing import preprocess_data
from RF.model import build_model, train_model
from RF.data_utils import prepare_data_for_rf, scale_data, split_train_test
from RF.visualization import plot_train_test_predictions
from RF.heatwave_predict import predict_heatwave
from RF.viza import draw_viz
from sklearn.preprocessing import MinMaxScaler


# Streamlit UI
st.title("Heatwave Prediction using RandomForest")
st.sidebar.header("Model Parameters")

# Load dataset
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = preprocess_data(uploaded_file)
    attr = df.drop(columns=['temp'])  # Features (exclude target column)
    tar = df['temp']
    st.write("### Sample Data")
    st.write(df.head())
    #df=preprocess_data(df)

    # Preprocess data
    #df = prepare_data_for_rf(df)
    scaler, scaled = scale_data(df)
    train, test = split_train_test(scaled)

    # Prepare data
    X_train, Y_train = train[:, :-1], train[:, -1]
    X_test, Y_test = test[:, :-1], test[:, -1]

    # Build and train the model
    n_estimators = st.sidebar.slider("Number of Estimators", min_value=50, max_value=500, step=50, value=100)
    model = build_model(n_estimators=n_estimators)

    st.sidebar.subheader("Training Model")
    if st.sidebar.button("Train Model"):
        rmse, mae, Y_pred, Y_real = train_model(model, X_train, Y_train, X_test, Y_test)
        st.markdown("---")
        st.write(f"### Model Evaluation")
        st.write(f"**Train Score: {rmse:.2f} RMSE**")
        st.write(f"**Test Score: {mae:.2f} MAE**")
        st.markdown("---")

        from sklearn.model_selection import train_test_split
        A,B,C,D= train_test_split(attr, tar, test_size=0.2, random_state=42)
        draw_viz(A, B, C, D)

        # Plot results
        date = df.tail(Y_real.shape[0]).index
        st.write("### Prediction vs Actual Data")
        fig = plot_train_test_predictions(Y_real, Y_pred, date)  # Get the figure from the plot function
        st.pyplot(fig)  

        st.markdown("---")
        # Predict heatwaves
        st.write(f"##Results")
        occurredHeatwave, predictedHeatwave, accuracy = predict_heatwave(X_train,Y_train,X_test, Y_test, Y_train, df)
        st.write(f"### Heatwave Prediction")
        st.write(f"**Occurred Heatwaves (2012-2022):** {occurredHeatwave}")
        st.write(f"**Predicted Heatwaves (2012-2022):** {predictedHeatwave}")
        st.write(f"**Heatwave Prediction Accuracy:** {accuracy*100:.2f}%")

