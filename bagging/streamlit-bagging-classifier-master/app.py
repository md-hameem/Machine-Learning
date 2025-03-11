import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# Set Seaborn style
sns.set(style='darkgrid')

# Generate synthetic data
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Function to create meshgrid for plotting decision boundaries
def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

# Sidebar configuration
st.sidebar.markdown("# Bagging Classifier")

# User inputs from sidebar
estimators = st.sidebar.selectbox('Select base estimator', ('Decision Tree', 'KNN', 'SVM'))
n_estimators = int(st.sidebar.number_input('Enter number of estimators', min_value=1, value=10))
max_samples = st.sidebar.slider('Max Samples', 0, 375, 375, step=25)
bootstrap_samples = st.sidebar.radio("Bootstrap Samples", ('True', 'False'))
max_features = st.sidebar.slider('Max Features', 1, 2, 2)
bootstrap_features = st.sidebar.radio("Bootstrap Features", ('False', 'True'))

# Plot initial scatter graph
fig, ax = plt.subplots()
ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
st.pyplot(fig)

# Button to run the algorithm
if st.sidebar.button('Run Algorithm'):
    # Select the base estimator
    if estimators == "Decision Tree":
        estimator = DecisionTreeClassifier()
        clf = estimator
    elif estimators == "KNN":
        estimator = KNeighborsClassifier()
        clf = estimator
    else:
        estimator = SVC()
        clf = estimator

    # Train the base estimator
    clf.fit(X_train, y_train)
    y_pred_tree = clf.predict(X_test)

    # Train Bagging Classifier
    bag_clf = BaggingClassifier(estimator=estimator,
                                n_estimators=n_estimators,
                                max_samples=max_samples,
                                bootstrap=bootstrap_samples == 'True',
                                max_features=max_features,
                                bootstrap_features=bootstrap_features == 'True',
                                random_state=42)
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)

    # Generate meshgrid and predictions for decision boundaries
    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)
    labels1 = bag_clf.predict(input_array)

    # Create subplots to display the results
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    # Plot the decision boundaries for both models
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax.set_title(f"{estimators} - Decision Boundary")
    st.pyplot(fig)

    ax1.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
    ax1.contourf(XX, YY, labels1.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    ax1.set_title("Bagging Classifier - Decision Boundary")
    st.pyplot(fig1)

    # Display accuracies
    st.subheader(f"Accuracy for {estimators}: {accuracy_score(y_test, y_pred_tree):.2f}")
    st.subheader(f"Accuracy for Bagging Classifier: {accuracy_score(y_test, y_pred):.2f}")
