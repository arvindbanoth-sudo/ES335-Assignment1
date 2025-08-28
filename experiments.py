import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 5  # repeat multiple times to smooth timings


# --------------------------------------------------------
# 1. Function to create fake binary dataset
# --------------------------------------------------------
def generate_fake_data(N, M):
    """
    Generate dataset with N samples and M binary features.
    Labels are random binary values.
    """
    X = pd.DataFrame(np.random.randint(0, 2, size=(N, M)), 
                     columns=[f"f{i}" for i in range(M)])  # make DataFrame with feature names
    y = pd.Series(np.random.randint(0, 2, size=N))        # make labels a Series
    return X, y


# --------------------------------------------------------
# 2. Function to measure average fit & predict time
# --------------------------------------------------------
def measure_times(N, M, criterion):
    """
    Measure average time for fitting and prediction for given N, M and criterion.
    """
    X, y = generate_fake_data(N, M)

    fit_times, pred_times = [], []

    for _ in range(num_average_time):
        clf = DecisionTree(criterion=criterion)

        # Measure fit time
        start = time.time()
        clf.fit(X, y)
        fit_times.append(time.time() - start)

        # Generate random test data (same shape as train for fairness)
        X_test, _ = generate_fake_data(N, M)

        # Measure predict time
        start = time.time()
        clf.predict(X_test)
        pred_times.append(time.time() - start)

    return np.mean(fit_times), np.std(fit_times), np.mean(pred_times), np.std(pred_times)


# --------------------------------------------------------
# 3. Run experiments for different N, M, and criteria
# --------------------------------------------------------
def run_experiments():
    Ns = [100, 500, 1000, 2000]   # vary number of samples
    Ms = [5, 10, 20, 50]          # vary number of features
    # ⚠️ Removed "random" because most DT implementations don’t support it
    criteria = ["information_gain", "gini_index", "misclassification_error"]

    results = {}

    for criterion in criteria:
        results[criterion] = {"fit": [], "predict": []}

        for N in Ns:
            for M in Ms:
                fit_mean, fit_std, pred_mean, pred_std = measure_times(N, M, criterion)
                results[criterion]["fit"].append((N, M, fit_mean, fit_std))
                results[criterion]["predict"].append((N, M, pred_mean, pred_std))

                print(f"{criterion} | N={N}, M={M} => fit: {fit_mean:.4f}s, predict: {pred_mean:.4f}s")

    return results, Ns, Ms


# --------------------------------------------------------
# 4. Plotting results
# --------------------------------------------------------
def plot_results(results, Ns, Ms):
    for criterion, data in results.items():
        # --- Fit time ---
        plt.figure(figsize=(10, 6))
        for M in Ms:
            fit_times = [t[2] for t in data["fit"] if t[1] == M]
            plt.plot(Ns, fit_times, marker='o', label=f"M={M}")
        plt.xlabel("Number of samples (N)")
        plt.ylabel("Fit Time (s)")
        plt.title(f"Fit Time vs N ({criterion})")
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- Predict time ---
        plt.figure(figsize=(10, 6))
        for M in Ms:
            pred_times = [t[2] for t in data["predict"] if t[1] == M]
            plt.plot(Ns, pred_times, marker='o', label=f"M={M}")
        plt.xlabel("Number of samples (N)")
        plt.ylabel("Predict Time (s)")
        plt.title(f"Predict Time vs N ({criterion})")
        plt.legend()
        plt.grid(True)
        plt.show()


# --------------------------------------------------------
# 5. Main
# --------------------------------------------------------
if __name__ == "__main__":
    results, Ns, Ms = run_experiments()
    plot_results(results, Ns, Ms)
