import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_x_new = diabetes_x[:,np.newaxis,2]
print('original data: ', diabetes_x[0:5,2])
print('reformatted data: ', diabetes_x_new[0:5])

diabetes_x_new_train1 = diabetes_x_new[:-20]
diabetes_x_new_test1 = diabetes_x_new[-20:]
diabetes_y_new_train1 = diabetes_y[:-20]
diabetes_y_new_test1 = diabetes_y[-20:]
print('length of training set ',len(diabetes_x_new_train1))
print('length of testing set ',len(diabetes_x_new_test1))

diabetes_x_new_train2 = diabetes_x_new[:-100]
diabetes_x_new_test2 = diabetes_x_new[-100:]
diabetes_y_new_train2 = diabetes_y[:-100]
diabetes_y_new_test2 = diabetes_y[-100:]
print('length of training set ',len(diabetes_x_new_train2))
print('length of testing set ',len(diabetes_x_new_test2))

regr1 = linear_model.LinearRegression()
regr1.fit(diabetes_x_new_train1,diabetes_y_new_train1)
print(regr1)

regr2 = linear_model.LinearRegression()
regr2.fit(diabetes_x_new_train2,diabetes_y_new_train2)
print(regr2)

diabetes_y_new_pred1 = regr1.predict(diabetes_x_new_test1)
print(diabetes_y_new_pred1)
diabetes_y_new_pred2 = regr2.predict(diabetes_x_new_test2)
print(diabetes_y_new_pred2)
#
# fig1,([ax1,ax2])=plt.subplots(1,2)
# ax1.scatter(diabetes_x_new_test1,diabetes_y_new_test1,color='black')
# ax1.scatter(diabetes_x_new_test1,diabetes_y_new_pred1,color='blue')
# ax1.set_title(['model with 20 sample'])
#
# ax2.scatter(diabetes_x_new_test2,diabetes_y_new_test2,color='black')
# ax2.scatter(diabetes_x_new_test2,diabetes_y_new_pred2,color='blue')
# ax2.set_title(['model with 100 sample'])
#
# plt.show();

# print('adf')
# print('Coefficients for model with testing sample of 20:{}',format(regr1.coef_))
# print('Mean squared error for model with testing sample of 20: %.2f' % mean_squared_error(diabetes_y_new_test1,diabetes_y_new_pred1))
# print('R-square for model with testing sample of 20: %.2f \n' % r2_score(diabetes_y_new_test1,diabetes_y_new_pred1))
#
# print('Coefficients for model with testing sample of 100:{}',format(regr2.coef_))
# print('Mean squared error for model with testing sample of 100: %.2f' % mean_squared_error(diabetes_y_new_test2,diabetes_y_new_pred2))
# print('R-square for model with testing sample of 100: %.2f \n' % r2_score(diabetes_y_new_test2,diabetes_y_new_pred2))

diabetes_x_df = pd.DataFrame(diabetes_x)
# print(diabetes_x_df.head())
diabetes_y_df = pd.DataFrame(diabetes_y)
# print(diabetes_y_df.head())

diabetes_x_df_train1 = diabetes_x_df.iloc[:-20,:]
diabetes_x_df_test1 = diabetes_x_df.iloc[-20:,:]
# print('Shape of train sample', diabetes_x_df_train1.shape)
# print('Shape of test sample', diabetes_x_df_test1.shape)
diabetes_y_df_train1 = diabetes_y_df.iloc[:-20]
diabetes_y_df_test1 = diabetes_y_df.iloc[-20:]

regr1_allfeature = linear_model.LinearRegression()
regr1_allfeature.fit(diabetes_x_df_train1,diabetes_y_df_train1)

diabetes_y_df_pred1 = regr1_allfeature.predict(diabetes_x_df_test1)
print(diabetes_y_df_pred1)
print('Coefficients for model with testing sample of 20:{}',format(regr1_allfeature.coef_))
print('Mean squared error for model with testing sample of 20: %.2f' % mean_squared_error(diabetes_y_df_test1,diabetes_y_df_pred1))
print('R-square for model with testing sample of 20: %.2f \n' % r2_score(diabetes_y_df_test1,diabetes_y_df_pred1))

RidgeCV_df = linear_model.RidgeCV()
# print(RidgeCV_df)
RidgeCV_df.fit(diabetes_x_df_train1,diabetes_y_df_train1)
diabetes_y_df_pred1_ridge = RidgeCV_df.predict(diabetes_x_df_test1)
# print(diabetes_y_df_pred1_ridge)
print('Coefficients for model with testing sample of 20 using ridge regression:{}',format(RidgeCV_df.coef_))
print('Mean squared error for model with testing sample of 20 using ridge regression: %.2f' % mean_squared_error(diabetes_y_df_test1,diabetes_y_df_pred1_ridge))
print('R-square for model with testing sample of 20 using ridge regression: %.2f \n' % r2_score(diabetes_y_df_test1,diabetes_y_df_pred1_ridge))

# data = {'area':[3456,2089,1416,5000,6325,3255,4255,6255],'price':[600,395,232,800,922,525,500,555]}
# df=pd.DataFrame(data=data)
# print(df)

# plt.scatter(df['area'],df['price'])
# plt.xlabel('area')
# plt.ylabel('price')
# plt.show()
