import streamlit as st
from RF.functions import train_random_forest, plot_performance, plot_feature_importance, plot_learning_curve, plot_cross_validation, plot_actual_vs_predicted, plot_error_distribution, plot_random_forest_tree, plot_calibration_curve

# Function to visualize with Streamlit
def draw_viz(X_train, X_test, y_train, y_test): 
    # Train model and get performance metrics
    model, mae_train, mae_test, r2_train, r2_test, y_train_pred, y_test_pred = train_random_forest(X_train, y_train, X_test, y_test)

    # Create performance plot
    st.subheader("Model Performance Comparison")
    performance_fig = plot_performance(mae_train, mae_test, r2_train, r2_test)
    st.pyplot(performance_fig)

    # Feature importance
    st.subheader("Feature Importance")
    feature_importance_fig = plot_feature_importance(model, X_train)
    st.pyplot(feature_importance_fig)

    # Learning curve
    st.subheader("Learning Curve")
    learning_curve_fig = plot_learning_curve(model, X_train, y_train)
    st.pyplot(learning_curve_fig)

    # Cross-validation results
    st.subheader("Cross-Validation Results")
    cross_val_fig = plot_cross_validation(model, X_train, y_train)
    st.pyplot(cross_val_fig)

    # Actual vs Predicted plot
    st.subheader("Actual vs Predicted")
    actual_vs_predicted_fig = plot_actual_vs_predicted(y_test, y_test_pred)
    st.pyplot(actual_vs_predicted_fig)

    # Error distribution plot
    st.subheader("Error Distribution")
    error_distribution_fig = plot_error_distribution(y_test, y_test_pred)
    st.pyplot(error_distribution_fig)

    # Random Forest Tree visualization
    st.subheader("Random Forest - Decision Tree Visualization")
    random_forest_tree_fig = plot_random_forest_tree(model, X_train)
    print("Plotting Decision Tree.....")
    st.pyplot(random_forest_tree_fig)

    

