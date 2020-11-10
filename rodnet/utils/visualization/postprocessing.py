import matplotlib.pyplot as plt


def visualize_ols_hist(olss_flatten):
    _ = plt.hist(olss_flatten, bins='auto')  # arguments are passed to np.histogram
    plt.title("OLS Distribution")
    plt.show()
