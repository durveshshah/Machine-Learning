import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt

print("...............Reading the Dataset  and Dataset Pre-Processing ................")
start_time = time.time()
#sigmoid activation function
def sigmoid(input,weight,bias):
  z = np.dot(input,weight)+bias
  return 1 / (1 + np.exp(-z))


def extract(file):
  with open(file, 'rb') as fo:
    load_pickle = pickle.load(fo, encoding='bytes')
  return load_pickle


#Load training data
train_list=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
train_to_append =[]
for i in train_list:
  # Creating Dataframe
  dataset = pd.DataFrame()

  # Extract data from the dataset
  extract_data = extract("D:/Machine learning Assignments/Assignment 2_part2/Data/cifar-10-batches-py/{}".format(i))
  extracted_list1 = list(extract_data.values())


  dataset["first_column"] = extracted_list1[1]

  # Parsing second column as list due to unsupported data format
  dataset["second_column"] = list(extracted_list1[2])


  # normalizing the data
  for j in range(0, len(dataset)):
    select_col = np.linalg.norm(dataset["second_column"][j])
    normalalized_data = dataset["second_column"][j] / select_col
    dataset["second_column"][j] = normalalized_data

  # Extract unique data from first column
  class_col = dataset["first_column"].unique()
  i = 0
  while i < len(class_col):
    dataset["x{}".format(class_col[i])] = np.where(dataset["first_column"].values == class_col[i], 1, 0)
    i += 1


  # shuffle the dataset
  dataset = dataset.sample(frac=1).reset_index(drop=True)

  # Append the Dataset to the list
  train_to_append.append(dataset)

# Concatenate the values to train data to be able to use below
train_data=pd.concat(train_to_append)


# Load Testing the data
test_to_append = []
# Creating Dataframe
df = pd.DataFrame()

# Extract data from the dataset
test_data_ = extract("D:/Machine learning Assignments/Assignment 2_part2/Data/cifar-10-batches-py/test_batch")

# Parsing second column as list due to unsupported data format
test_extracted_list = list(test_data_.values())


df["first_column"] = test_extracted_list[1]
df["second_column"] = list(test_extracted_list[2])

# normalizing the data
for i in range(0, len(df)):
  select_col = np.linalg.norm(df["second_column"][i])
  normalalized_data = df["second_column"][i] / select_col
  df["second_column"][i] = normalalized_data

# Extract unique data from first column
class_col = df["first_column"].unique()
i = 0
while i < len(class_col):
  df["x{}".format(class_col[i])] = np.where(df["first_column"].values == class_col[i], 1, 0)
  i+=1

# shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Append the Dataset to the list
test_to_append.append(df)

# Concatenate the values to train data to be able to use below
test_data=pd.concat(test_to_append)

# Giving number of epochs i.e. hidden number of neurons to run
epochs = 500

end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Pre-processing and Reading the Dataset: %f seconds \n " % total_time)

# Method to process Incremental extreme machine learning
print(".................Training model...................")
start_time = time.time()
def train(top_acc,training_data,no_of_neurons):

  # Assigning weights,bias and beta to zeros array
  weights_to_copy = np.zeros(no_of_neurons)
  bias_to_copy = np.zeros(no_of_neurons)
  beta_to_copy = np.zeros(no_of_neurons)

  # Taking second column and converting it to a list because of unsupported data
  df= training_data["second_column"]
  df = df.tolist()

  # Making an array and putting in unique data that we extracted earlier
  actual=training_data[['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']]
  actual = actual.to_numpy()

  # making a list to append the accuracy
  acc_to_append=[]
  # Iterate over number of neurons
  for i in range(1,no_of_neurons):

    # We need size so as to update neurons for each iteration
    bias = np.random.normal(size=[i])
    input_weights = np.random.normal(size=[3072,i])

    # Activation function
    input=sigmoid(df,input_weights,bias)

    # Transpose input activation function
    transpose_input =np.transpose(input)

    # Calculating beta
    beta_param = np.dot(np.linalg.inv(np.dot(transpose_input,input)),np.dot(transpose_input,actual))

    # Getting dot product of input and beta
    pred_out =np.dot(input,beta_param)

    # calculating the counter to calculate the accuracy.
    # Normal for loop was taking a lot of time so tried using list comprehension
    c = [np.dot(z,w) for (z, w) in zip(actual,np.where(pred_out>0.5,1,0))].count(1)
    accuracy=(c/len(actual))*100


    # Copy the values if top acuuracy is less than accuracy so that it can be used later
    if top_acc < accuracy:
      top_acc=accuracy
      beta_to_copy=beta_param
      bias_to_copy=bias
      weights_to_copy=input_weights

    acc_to_append.append(accuracy)


  return top_acc,weights_to_copy,bias_to_copy,beta_to_copy,acc_to_append


# initialize greatest_acc to zero just before calling the function to avoid the shape error.
greatest_acc = 0
top_acc,weights_to_copy,bias_to_copy,beta_to_copy,acc_to_append=train(greatest_acc,train_data,epochs)

end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Training: %f seconds \n " % total_time)

print("Training Accuracy ",top_acc)

print("......................Testing...........................")
start_time = time.time()
# Perform the same procedure as training however since this is only testing we don't calculate beta. We just pass our test data
test_frame = test_data["second_column"]
test_frame = test_frame.tolist()
actual_test = test_data[['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']]
actual_test = actual_test.to_numpy()
test_input = sigmoid(test_frame, weights_to_copy, bias_to_copy)
pred_out_test = np.dot(test_input, beta_to_copy)
pred_out_test = np.where(pred_out_test > 0.5, 1, 0)


c = []
count = 1
for z,w in zip(actual_test, pred_out_test):
  c.append(np.dot(z,w))
  count = c.count(1)
accuracy = (count / len(actual_test)) * 100

end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Testing: %f seconds \n " % total_time)

print("Testing Accuracy ",accuracy)

print("......................Plotting Graph...........................")
start_time = time.time()
plt.figure()
plt.xlabel("Number of hidden neurons {}".format(epochs))
plt.title("Classification Learning curve")
plt.ylabel("Accuracy")
plt.plot(acc_to_append,color="green")
plt.show()

end_time = time.time()
total_time = end_time - start_time
print("Time Cost for Testing: %f seconds \n " % total_time)

