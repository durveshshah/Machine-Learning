import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

'''
This Back-propagation algorithm is implemented on Combined Cycle Power Plant UCI Dataset. Link is given below
Dataset Link: https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant
'''
print("...............Reading the Dataset  and Dataset Pre-Processing ................")
start_time = time.time()
dataset = shuffle(pd.read_csv("ccip_dataset.csv"))



x = dataset[["AT","V","AP","RH"]]
y = dataset[['PE']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)


# Normalizing data using Standard Scaler Fit Transform
x_train = pre.StandardScaler().fit_transform(x_train)
x_test = pre.StandardScaler().fit_transform(x_test)

# Converting pd dataframe to numpy array to match compatibility
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Pre-processing and Reading the Dataset: %f seconds \n " % total_time)


# Hyperbolic Tangent Activation function
def hyperbolic_tanh(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

# Hyperbolic derivative
def derivative_hyperbolic(x):
  return 1 - hyperbolic_tanh(x) * hyperbolic_tanh(x)

print("............... Initializing hyperparameters ................")
start_time = time.time()
# Setting Hyperparameters
actual_out_size = y_train.size
np.random.seed(10)
inp = 4
hd = 6
out = 1

print("............... Setting 4 weights for hidden layers ................")
w1_l1 = np.random.randn(inp, hd)
w2_l2 = np.random.randn(hd, hd)
w3_l3 = np.random.randn(hd, hd)
w4_l4 = np.random.randn(hd, hd)
out_w = np.random.randn(hd, out)

rmse_list = []

epochs = 500
eta = 0.0001
alpha = 0.7

end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Setting HyperParameters: %f seconds \n " %total_time)

print("............... Training Backpropagation Algorithm ................")
start_time = time.time()
for epoch in range(epochs):
    # Feedforward for 4 hidden layers by calling activation function
    l1 = np.dot(x_train, w1_l1)
    l1_out = hyperbolic_tanh(l1)

    l2 = np.dot(l1_out, w2_l2)
    l2_out = hyperbolic_tanh(l2)

    l3 = np.dot(l2_out, w3_l3)
    l3_out = hyperbolic_tanh(l3)

    l4 = np.dot(l3_out, w4_l4)
    l4_out = hyperbolic_tanh(l4)

    output = np.dot(l4_out, out_w)
    final_out = hyperbolic_tanh(output)

    rmse = np.sqrt(np.mean(np.square(final_out - y_train))) /100
    rmse_list.append(rmse)

    # Backpropagation for 4 hidden layers
    final_err = final_out - y_train
    final_tanh_derivative = final_err * derivative_hyperbolic(final_out)

    l4_err = np.dot(final_tanh_derivative, out_w.T)
    l4_derivative = l4_err * derivative_hyperbolic(l4_out)

    l3_err = np.dot(l4_derivative, w4_l4.T)
    l3_derivative = l3_err * derivative_hyperbolic(l3_out)

    l2_err = np.dot(l3_derivative, w3_l3.T)
    l2_derivative = l2_err * derivative_hyperbolic(l2_out)

    l1_err = np.dot(l2_derivative, w2_l2.T)
    l1_derivative = l1_err * derivative_hyperbolic(l1_out)

    # Divide weights as per size of output
    output_weights = np.dot(l4_out.T, final_tanh_derivative) / actual_out_size
    weights4 = np.dot(l3_out.T, l4_derivative) / actual_out_size
    weights3 = np.dot(l2_out.T, l3_derivative) / actual_out_size
    weights2 = np.dot(l1_out.T, l2_derivative) / actual_out_size
    weights1 = np.dot(x_train.T, l1_derivative) / actual_out_size

    out_w -= eta * alpha * output_weights
    w4_l4 -= eta * alpha * weights4
    w3_l3 -= eta * alpha * weights3
    w2_l2 -= eta * alpha * weights2
    w1_l1 -= eta * alpha * weights1

print("Training RMSE: "+str(round(rmse_list[-1],2)))
end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Training algorithm: %f seconds \n " %total_time)

print("............... Plotting RMSE Curve ................")
plt.title("RMSE Curve")
plt.ylabel("RMSE")
plt.xlabel("Epochs")
plt.plot(rmse_list)
plt.show()


print()
print("............... Testing Backpropagation Algorithm ................")
start_time = time.time()

# Feedforward for 4 hidden layers by calling activation function
l1 = np.dot(x_test, w1_l1)
l1_out = hyperbolic_tanh(l1)

l2 = np.dot(l1_out, w2_l2)
l2_out = hyperbolic_tanh(l2)

l3 = np.dot(l2_out, w3_l3)
l3_out = hyperbolic_tanh(l3)

l4 = np.dot(l3_out, w4_l4)
l4_out = hyperbolic_tanh(l4)

output = np.dot(l4_out, out_w)
final_out = hyperbolic_tanh(output)

# Calculate RMSE
rmse = np.sqrt(np.mean(np.square(final_out - y_test))) /100

print("Testing RMSE: "+str(round(rmse,2)))
end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Testing algorithm: %f seconds \n " %total_time)

print(final_out)