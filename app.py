
import numpy as np
import math
import sklearn
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.svm import SVR, SVC



file_name = ["power_dataset.csv"]

data = []

for f in file_name:
    fil = open(f,"r")
    rl = fil.readlines()
    rl.pop(0)
    tem_1 = []
    for l in rl:
        sp = l.split(',')
        sp.pop(0)
        sp.pop(3)
        sp = sp[:5]
        tem_2 = []

        for p in sp:
            if p == "":
                tem_2 += [float(0)]
            else:
                tem_2 += [float(p)]
        tem_1 += [tem_2]
    data += tem_1


data = np.array(data)
print("all data shape = ", data.shape)
label = data[:,1]
data = np.delete(data, 1, 1)

is_normal = 1

#normalize
if  is_normal == 1:
    data_max = np.max(data, axis = 0)
    data_min = np.min(data, axis = 0)
    data = (data-data_min)/(data_max-data_min)
    label_max = np.max(label, axis = 0)
    label_min = np.min(label, axis = 0)
    label = (label-label_min)/(label_max-label_min)

test_interval = 67
label_interval = 0
test_ratio = 0.00

data_tem_size = data.shape[0] - test_interval - label_interval
train_data = np.zeros((data_tem_size, data.shape[1]))
if label_interval == 0:
    train_label = np.zeros((data_tem_size))
else:
    train_label = np.zeros((data_tem_size, label_interval))

for i in range(data.shape[0] - test_interval - label_interval):
    train_data[i] = data[i]
    if label_interval == 0:
        train_label[i] = label[i+test_interval]
    else:
        train_label[i] = label[i+test_interval : i+test_interval+label_interval]

size = train_data.shape[0]-int(data_tem_size * test_ratio)

test_data = train_data[size:]
test_label = train_label[size:]
train_data = train_data[:size]
train_label = train_label[:size]

print("train data shape = ", train_data.shape)
print("train label shape = ", train_label.shape)
print("test data shape = ", test_data.shape)
print("test label shape = ", test_label.shape)

model = SVR(kernel = "rbf", gamma = 'scale', C = 200, max_iter = -1, tol = 1e-1, epsilon = 1e-2)

print(train_data.shape, train_data.dtype)
print(train_label.shape, train_label.dtype)

model.fit(train_data, train_label )

def rmse(x, y):
    return math.sqrt(np.mean(np.square(x-y)))

if int(data_tem_size * test_ratio) != 0:
    test_pre = model.predict(test_data)

    if is_normal == 1:
        n_test_data = (test_data*(data_max-data_min) + data_min)
        n_test_label = (test_label*(label_max-label_min) + label_min)
        n_test_pre = (test_pre*(label_max-label_min) + label_min)
    else:
        n_test_data = test_data
        n_test_label = test_label
        n_test_pre = test_pre
    
    print("data")
    print(n_test_data)
    print("ground_label")
    print(n_test_label)
    print("predict_label")
    print(n_test_pre)
    
    test_loss = np.zeros(test_label.shape[0])
    for i in range(test_label.shape[0]):
        test_loss = rmse(n_test_label[i], n_test_pre[i])
    
    print("\nloss : ", rmse(n_test_label, n_test_pre), "\n")


output_data = data[-7:]
output_label = model.predict(output_data)
if is_normal == 1:
    n_output_data = (output_data*(data_max-data_min) + data_min)
    n_output_label = (output_label*(label_max-label_min) + label_min)
else:
    n_output_data = output_data
    n_output_label = output_label
print("output data = ", n_output_data)
print("output label = ", n_output_label)

is_output = 1
if is_output == 1:
    i = 0
    while(i < 100):
        if(i == 0):
            name = "submission.csv"
        else:
            name = "submission_" + str(i) + ".csv"
        try:
            out = open(name, "r") 
        except:
            out = open(name, "w") 
            out.write("date,peak_load(MW)\n")
            for j in range(n_output_label.shape[0]):
                output_str = "2019040" + str(j+2) + "," + str(n_output_label[j]) + "\n"
                out.write(output_str)
            out.close()
            break
        else:
            out.close()
            i += 1

