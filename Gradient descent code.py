import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\amitf\\Desktop\\amit\\למידת מכונה\\מטלה 1\\sample.csv")

def costs(a, b, x, y):
    y_pred = a * x + b
    cost = np.sum(np.square(y_pred-y))
    return cost
#-----------------------------------------------------------------------------
#gradient_descent
a,b,epochs= 0,0,1000
learning_rate = 0.0001#0.1
x,y = data['x'].values,data['y'].values
lisa ,lisb , liscost = [],[],[]
def gradient_descent(x, y, a, b, learning_rate, epochs):
    for i in range(epochs):
        y_pred = a * x + b
        error = y - y_pred
        Dm = -(2/len(y)) * np.sum(error * x)
        Dc = -(2/len(y)) * np.sum(error)
        a = a - learning_rate * Dm
        b = b - learning_rate * Dc
        cost = costs(a, b, x, y)
        lisa.append(a) ,lisb.append(b) , liscost.append(cost)
    return a, b, lisa, liscost, lisb

a, b, cost_history, a_history, b_history = gradient_descent(x, y, a, b, learning_rate, epochs)
print('a: {:.2f}, b: {:.2f}'.format(a, b))
plt.plot(range(epochs), liscost)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Gradient Descent - Cost History')
plt.show()

plt.plot(range(epochs), lisa, label='a')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Gradient Descent - Values of a')
plt.legend()
plt.show()

plt.plot(range(epochs), lisb, label='b')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Gradient Descent - Values of b')
plt.legend()
plt.show()

#-----------------------------------------------------------------------------
#Stochastic gradient descent
a,b,epochs= 0,0,1000
learning_rate = 0.1#0.0001
x,y = data['x'].values,data['y'].values
lisa ,lisb , liscost = [],[],[]
def SGD(a , b , x, y):
    y_pred = a * x + b
    error = y - y_pred
    Dm = -(2/len(y_shuffled)) * np.sum(error * x)
    Dc = -(2/len(y_shuffled)) * np.sum(error)
    a = a - learning_rate * Dm
    b = b - learning_rate * Dc
    cost = costs(a, b, x, y)
    lisa.append(a)
    lisb.append(b)
    liscost.append(cost)
    return a,b,cost


for i in range(1000):
    idx = np.random.permutation(len(y))
    x_shuffled , y_shuffled= x[idx],y[idx]
    j = np.random.randint(len(y))
    a , b, cost = SGD(a , b , x_shuffled[j], y_shuffled[j])



# Print the learned parameters
print('a: {:.2f}, b: {:.2f}, cost: {:.2f}'.format(a, b,cost))
plt.plot(range(len(liscost)), liscost)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('SGD - Cost History')
plt.show()

# Plot the values of 'a' and 'b'
plt.plot(range(len(lisa)),lisa, label='a')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('SGD - Values of a')
plt.legend()
plt.show()

plt.plot(range(len(lisb)),lisb, label='b')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('SGD - Values of b')
plt.legend()
plt.show()

#-----------------------------------------------------------------------------
#Mini Batch Stochastic gradient descent
a,b,epochs= 0,0,1000
learning_rate = 0.1#0.0001
x,y = data['x'].values,data['y'].values
lisa ,lisb , liscost = [],[],[]
def MBSGD(a , b , x, y):
    y_pred = a * x + b
    error = y - y_pred
    Dm = -(2/len(y_shuffled)) * np.sum(error * x)
    Dc = -(2/len(y_shuffled)) * np.sum(error)
    a = a - learning_rate * Dm
    b = b - learning_rate * Dc
    cost = costs(a, b, x, y)
    lisa.append(a)
    lisb.append(b)
    liscost.append(cost)
    return a,b,cost

for i in range(1000):
    idx = np.random.permutation(len(y))
    x_shuffled = x[idx]
    y_shuffled = y[idx]
    j = np.arange(8)
    a , b, cost = MBSGD(a , b , x_shuffled[j], y_shuffled[j])



# Print the learned parameters
print('a: {:.2f}, b: {:.2f}, cost: {:.2f}'.format(a, b,cost))
plt.plot(range(len(liscost)), liscost)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('MBSGD - Cost History')
plt.show()

# Plot the values of 'a' and 'b'
plt.plot(range(len(lisa)),lisa, label='a')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('MBSGD - Values of a')
plt.legend()
plt.show()

plt.plot(range(len(lisb)),lisb, label='b')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('MBSGD - Values of b')
plt.legend()
plt.show()
