import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
datafile= pd.read_csv("data/house_price.csv")

# print(datafile)

x_dataset= datafile.iloc[:,0:2].values.astype(float)


# Scaling
# x_dataset[:, 0]= x_dataset[:,0]/ 4215
# x_dataset[:, 1]= x_dataset[:,1]/ 5


#Z-Score scaling
mean_x= np.mean(x_dataset,axis=0)
std_x= np.std(x_dataset, axis=0)
x_dataset= (x_dataset- mean_x)/std_x 

y_dataset= datafile.iloc[:, -1].values.astype(float)
mean_y = np.mean(y_dataset)
std_y = np.std(y_dataset)
y_dataset = (y_dataset - mean_y) / std_y
print(x_dataset[0,0])

w= np.zeros(2)
b= 0

# print(x_dataset.shape[0])

# print(x_dataset[0].shape[0])
# print(y_dataset)


x1= np.array([[1,2,3],[2,3,4]])
y1= np.array([1,2,3])


yb= np.dot(x1,y1)
# print(x1[1])


# 

def cost_function(x_dataset, y_dataset, w, b):

    total_cost=0


    m= x_dataset.shape[0]
    total_cost= 0.0
    j_temp= 0.0

    
    for i in range(m):
        f_wb= np.dot(x_dataset[i],w) + b
        j_temp= j_temp + ((f_wb- y_dataset[i])**2)


    total_cost= (1/(2*m))* j_temp
    return total_cost


def gradient(x_dataset, y_dataset, w , b):
     m= x_dataset.shape[0]
     j= x_dataset[0].shape[0]


     dj_dw=np.zeros(j)
     dj_db=0.0


     for i in range(m):
          f= (np.dot(x_dataset[i],w) + b)- y_dataset[i]
          for l in range(j):
               dj_dw[l]= dj_dw[l]  + f*x_dataset[i][l]

          dj_db= dj_db + f

     dj_dw= (1/m)*dj_dw
     dj_db= (1/m)* dj_db
     
     return dj_dw, dj_db




def gradient_descent(x_dataset, y_dataset, w_in , b_in, alpha, num_iterations):
     temp_w= np.zeros(x_dataset[0].shape[0])
     temp_b= 0.0
     

     w= copy.deepcopy(w_in)
     b= b_in
     n= x_dataset[0].shape[0]
     dj= x_dataset[0].shape[0]
     db=0.0

     cost_history = []
     w1_history = []
 
     for i in range(num_iterations):
          
          dj, db= gradient(x_dataset,y_dataset,w,b)
          cost = cost_function(x_dataset, y_dataset, w, b)
          cost_history.append(cost)
          w1_history.append(w[0])
          temp_b= b- (alpha*db)
          b= temp_b
        
          for l in range(n):
               temp_w[l]= w[l]- (alpha*dj[l])
               w[l]= temp_w[l]

               
     return w,b, cost_history, w1_history




# result= gradient_descent(x_dataset, y_dataset, w, 0, 0.01, 10000)
# print(result)

w, b, cost_history, w1_history = gradient_descent(
    x_dataset, y_dataset, w, 0, 0.01, 10000
)


plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs Iteration")
plt.show()


plt.plot(w1_history, cost_history)
plt.xlabel("w1")
plt.ylabel("Cost")
plt.title("Cost vs w1 (during training)")
plt.show()

          
     
               

y_pred= np.dot(x_dataset, w) + b
y_pred_real = y_pred * std_y + mean_y
y_real = y_dataset * std_y + mean_y

plt.scatter(y_real, y_pred_real)
plt.plot(y_real, y_real, color='red')
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted(Real Prices)")
plt.show()



