import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# Set plotting style (use 'seaborn' instead of 'seaborn-bright')
plt.style.use('seaborn')

# Constants
n_train = 150
n_test = 100
noise = 0.1

# Seed for reproducibility
np.random.seed(0)

# Generate data function
def f(x):
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5  # Generate X values
    X = np.sort(X).ravel()  # Sort the values
    y = f(X) + np.random.normal(0.0, noise, n_samples)  # Generate noisy Y values
    X = X.reshape((n_samples, 1))  # Reshape X to 2D
    return X, y

# Data generation
X_train, y_train = generate(n_samples=n_train, noise=noise)
X_test, y_test = generate(n_samples=n_test, noise=noise)

# Streamlit sidebar components
st.sidebar.markdown("# Bagging Regressor")

estimator = st.sidebar.selectbox('Select base estimator', ('Decision Tree', 'SVM', 'KNN'))
n_estimators = int(st.sidebar.number_input('Enter number of estimators', min_value=1, max_value=500, value=50))
max_samples = st.sidebar.slider('Max Samples', 0, n_train, n_train, step=25)
bootstrap_samples = st.sidebar.radio("Bootstrap Samples", ('True', 'False'))

# Initial graph with training data
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color="yellow", edgecolor="black")
st.pyplot(fig)

# Run Algorithm button
if st.sidebar.button('Run Algorithm'):
    # Select the base estimator and create the corresponding regressor
    if estimator == 'Decision Tree':
        base_model = DecisionTreeRegressor()
    elif estimator == 'SVM':
        base_model = SVR()
    else:
        base_model = KNeighborsRegressor()

    # Train the base model
    reg = base_model.fit(X_train, y_train)

    # Create Bagging Regressor
    bag_reg = BaggingRegressor(base_model, n_estimators=n_estimators, 
                               max_samples=max_samples, bootstrap=bootstrap_samples).fit(X_train, y_train)

    # Predictions
    bag_reg_predict = bag_reg.predict(X_test)
    reg_predict = reg.predict(X_test)

    # R2 Scores
    bag_r2 = r2_score(y_test, bag_reg_predict)
    reg_r2 = r2_score(y_test, reg_predict)

    # Empty initial plot and display results
    st.empty()

    # Plot Bagging Regressor result
    fig1, ax1 = plt.subplots()
    st.subheader(f"Bagging - {estimator} (R2 score - {bag_r2:.2f})")
    ax1.scatter(X_train, y_train, color="yellow", edgecolor="black")
    ax1.plot(X_test, bag_reg_predict, linewidth=2, label="Bagging")
    ax1.legend()
    st.pyplot(fig1)

    # Plot Base Estimator result
    fig2, ax2 = plt.subplots()
    st.subheader(f"{estimator} (R2 score - {reg_r2:.2f})")
    ax2.scatter(X_train, y_train, color="yellow", edgecolor="black")
    ax2.plot(X_test, reg_predict, linewidth=2, color='red', label=estimator)
    ax2.legend()
    st.pyplot(fig2)
