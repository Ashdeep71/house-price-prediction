import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



dataset= pd.read_csv("data/house_price.csv")
# print(dataset)
x_dataset= dataset.iloc[:, 0:2].values
y_dataset= dataset.iloc[:, -1].values

# print(x_dataset)

reg = LinearRegression()
reg.fit(x_dataset, y_dataset, sample_weight=None)

pred= reg.predict(x_dataset)





# print(pred)
print("Weights:", reg.coef_)
print("Bias:", reg.intercept_)

plt.scatter(y_dataset, pred, alpha=0.7)
plt.plot(y_dataset,y_dataset,color= "red")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted")
plt.show()


mse = np.mean((pred - y_dataset)**2)
print("Sklearn MSE:", mse)
 