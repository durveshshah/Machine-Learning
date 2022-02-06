'''
UCI Dataset Link: https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant
'''
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

print("...............Reading the Dataset  and Dataset Pre-Processing ................")
start_time = time.time()
# Read dataset
dataset = pd.read_csv("D:/Machine learning Assignments/Assignment 2_part2/CCPP/ccip_dataset.csv")

# normalizing the data
normalize_data = ["AT", "V", "AP", "RH", "PE"]
dataset[normalize_data] = dataset[normalize_data].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Splitting dataset into 80% training and 20% testing
dataset = dataset.sample(frac=1).reset_index(drop=True)
data_size = int(0.8 * len(dataset))
training_data = dataset[:data_size]
testing_data = dataset[data_size:]

# Shuffling data
shuffle_data_train = training_data.sample(frac=1).reset_index(drop=True)
shuffle_data_test = testing_data.sample(frac=1).reset_index(drop=True)

# Taking training input and outputs
train_set_input = shuffle_data_train[["AT", "V", "AP", "RH", "PE"]]
train_set_output = shuffle_data_train["PE"].to_numpy()

# Taking testing input and outputs
test_set_output = shuffle_data_test["PE"].to_numpy()
test_set_input = shuffle_data_test[["AT", "V", "AP", "RH", "PE"]]

# Converting dataset into array
dataset = np.array(dataset)




# sigmoid activation function
def sigmoid(input, weight, bias):
    z = np.dot(input, weight) + bias
    return 1 / (1 + np.exp(-z))

end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Pre-processing and Reading the Dataset: %f seconds \n " % total_time)

print(".................Training model...................")
start_time = time.time()

weight_for_test = 0
bias_for_test = 0
beta_for_test = 0
error = 1
epochs = 50
rmse_list = []  # list for rmse error
input_columns_size = np.size(dataset, 1)  # Calculate size of input columns
rmse = 0

for i in range(1, epochs + 1):

    # Adding neurons for each iteration for bias and weights
    bias = np.random.normal(size=[i])
    weights = np.random.normal(size=[input_columns_size, i])

    for j in range(1, len(training_data)):

        # calling activation function
        sigmoid_func = sigmoid(train_set_input, weights, bias)

        # Transposing data
        Transpose_sigmoid = sigmoid_func.T

        # We need to inverse data to calculate beta value. This is knows as H inverse
        beta = np.dot(np.linalg.inv(np.dot(Transpose_sigmoid, sigmoid_func)), np.dot(Transpose_sigmoid, train_set_output))

        # Calculate RMSE Error
        rmse = np.sqrt(np.mean(np.square(train_set_output - np.dot(sigmoid_func, beta))))

        # Append error to list so that graph can be plotted
        rmse_list.append(rmse)

        train_set_output = train_set_output - np.dot(sigmoid_func, beta)

        if rmse < error:
            error = rmse
            weight_for_test = weights
            bias_for_test = bias
            beta_for_test = beta


print("RMSE error for Training", rmse)


end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Training: %f seconds \n " % total_time)

print("......................Testing...........................")
start_time = time.time()
rmse_test = 0
for i in range(1, len(testing_data)):
    x_test = sigmoid(test_set_input, weight_for_test, bias_for_test)
    y_test = np.dot(x_test, beta_for_test)

    # Calculating RSME Error
    rmse_test = np.sqrt(np.mean(np.square(test_set_output - y_test)))

print("RMSE error for Testing", rmse_test)

plt.figure()
plt.xlabel("Number of iterations {}".format(epochs))
plt.title("Regression Learning curve")
plt.ylabel("RMSE Error")
plt.plot(rmse_list, color="green")
plt.show()

end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Testing: %f seconds \n " % total_time)
