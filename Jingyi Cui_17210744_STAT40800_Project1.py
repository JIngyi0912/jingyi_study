
# coding: utf-8




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
import sklearn.preprocessing as skp 
import matplotlib.pyplot as plt 
import random
import math
import itertools
import re
from mpl_toolkits.mplot3d import Axes3D


dataset = pd.read_csv('Project 1 - Dataset.csv')
dataset.head()
np.set_printoptions(suppress=True)


data_input = dataset[['Weight lbs','Height inch','Neck circumference','Chest circumference','Abdomen  circumference']]
data_output = dataset[['bodyfat']]
#import data



X = data_input
y = data_output


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
y_data = np.array(y_train)


normalized_dataset = skp.normalize(X_train, norm='max')
y_norm = skp.normalize(y_train,norm='max')
#Normalize the training set


def eq1_1(weights,data):
    result1 =[]
    result2=[]
    for i in range(len(data)):
        temp1 = data[i]
        temp2 =weights[:,i]*temp1
        result1.append(temp2)
    
    result1 = np.array(result1)
    for i in range(len(result1[0])):
        temp2 = sum(result1[:,i])
        result2.append(temp2)
    return result2
#first part of eq1


def eq1(data):
    result=[]
    for i in range(len(data)):
        temp1 = data[i]
        temp2=1.0/(1+math.e**-temp1)
        result.append(temp2)
    return result
#final eq1


def eq2(y_hat,y_data):
    tem2 = 0
    for i in range(len(y_hat)):
        tem1 = (y_hat[i]-y_norm[i])*(y_hat[i]-y_norm[i])
        tem2 = tem2 + tem1
    tem3 = (1-tem2/len(y_data))*100
    return tem3
#eq2


fitness_value = []
weights = []
for i in range(500):
    temp_weights = -1 + 2 * np.random.random((10,5))
    y_hat = []
    for j in range(len(normalized_dataset)):
        temp_result1 = eq1_1(temp_weights,normalized_dataset[j])
        temp_result2 = eq1(temp_result1)
        temp_hat = sum(temp_result2)
        y_hat.append(temp_hat)
        temp_fitness_value = eq2(y_hat,y_data)
    fitness_value.append(temp_fitness_value)
    weights.append(temp_weights)
#use loop to get 500 original weights


first_highest_fitness_value = max(fitness_value)
standard = first_highest_fitness_value
first_highest_location = fitness_value.index(first_highest_fitness_value)
sire = weights[first_highest_location]
#get the first highest fitness value and sire(first parent)


def own_scaler(weights):
    scaler = skp.MinMaxScaler()
    scaler.fit(weights)
    scaler.data_max_
    weights_normalized = scaler.transform(weights)
    b_weights = np.trunc(weights_normalized * 1000)
    return b_weights
#normalize weights


def binary(weight):
    return bin(int(weight))[2:].zfill(10)
#function for binary


def cross_over(chromo1,chromo2):
    site =  np.random.randint(2,len(chromo1)-1)
    child_chromosome1=(chromo1[:site]+chromo2[site:])
    child_chromosome2=(chromo1[site:]+chromo2[:site])
    return child_chromosome1,child_chromosome2
#function for cross over


def mutation(chromosome):
    site = np.random.randint(0,len(chromosome)-1)
    tem1 = list(chromosome)
    if tem1[site] != '0':
        tem1[site] = '0'
    else:
        tem1[site]= '1'
    new_chromosome = ''.join(tem1)
    return new_chromosome
#function for mutation


def back_weights(chromosome):
    tem1 = re.findall(r'.{10}',chromosome)
    tem2 = np.array(tem1)
    tem2.resize(10,5)
    return tem2
#function for de-segment chromosome


def decimal(weights):
    decimal_parent = []
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            temp = int(weights[i][j],2)
            decimal_parent.append(temp)
    temp1 = np.array(decimal_parent)
    temp1.resize(10,5)
    return temp1
#function for decimal


def de_normalized(weights):
    temp1 = weights/1000
    temp2 = skp.normalize(temp1, norm='max')
    temp3 = 2* temp2 - 1
    return temp3
#function for de-normalize weights


final_fitness = []
for i in range(60):
    all_weights = []
    all_fitness_value = []
    for i in range(len(weights)):
        parent1 = sire
        parent2 = weights[i]
        scaler_parent1 = own_scaler(parent1)
        scaler_parent2 = own_scaler(parent2)
        binary_tem1 = []
        binary_tem2 = []
        for i in scaler_parent1:
            binary_tem1.append(list(map(binary,i)))
        for j in scaler_parent2:
            binary_tem2.append(list(map(binary,j)))
        chrom1 = ''
        chrom2 = ''
        for i in binary_tem1:
            tem1 = ''.join(i)
            chrom1 = chrom1 + tem1
        for j in binary_tem2:
            tem2 = ''.join(j)
            chrom2 = chrom2 + tem2
        child_chromosome1,child_chromosome2 = cross_over(chrom1,chrom2)
        new_chrom1 = mutation(child_chromosome1)
        new_chrom2 = mutation(child_chromosome2)
        new_temp_weights1 = back_weights(new_chrom1)
        new_temp_weights2 = back_weights(new_chrom2)
        new_weights1 = decimal(new_temp_weights1)
        new_weights2 = decimal(new_temp_weights2) 
        final_weights1 = de_normalized(new_weights1)
        final_weights2 = de_normalized(new_weights2)
        all_weights.append(final_weights1)
        all_weights.append(final_weights2)
        y_hat = []
        for j in range(len(normalized_dataset)):
            temp_result1 = eq1_1(final_weights1,normalized_dataset[j])
            temp_result2 = eq1(temp_result1)
            temp_hat = sum(temp_result2)
            y_hat.append(temp_hat)
            temp_fitness_value1 = eq2(y_hat,y_data)
        all_fitness_value.append(temp_fitness_value1)
        y_hat = []
        for j in range(len(normalized_dataset)):
            temp_result1 = eq1_1(final_weights2,normalized_dataset[j])
            temp_result2 = eq1(temp_result1)
            temp_hat = sum(temp_result2)
            y_hat.append(temp_hat)
            temp_fitness_value2 = eq2(y_hat,y_data)
    all_fitness_value.append(temp_fitness_value2)
    new_highest_fitness_value = max(all_fitness_value)
    final_fitness.append(new_highest_fitness_value)
    print(new_highest_fitness_value)
    location = all_fitness_value.index(new_highest_fitness_value)
    if new_highest_fitness_value > standard:
        standard = new_highest_fitness_value
        sire = all_weights[location]
        if location >= len(all_weights):
            weights = all_weights[:location]
        else:
             weights = all_weights[:location]
#use loop to do cross over,mutation,and other operation,finally get the new highest fitness value and new weights 


fig = plt.figure()
x = final_fitness
y = x
plt.scatter(x,y)
plt.show
#scatter plot the highest fitness value for each iteration


my_weights = all_weights[location]


test_output = skp.normalize(y_test,norm='max')
test_input = skp.normalize(X_test,norm='max')



my_y_hat = []
for i in range(len(test_input)):
    result_a = eq1_1(my_weights,test_input[i])
    result_b = eq1(result_a)
    result_hat = sum(result_b)
    my_y_hat.append(result_hat)
    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
z1 = my_y_hat
z2 = test_output
x = test_input[:,1]
y = test_input[:,2]
ax.scatter(x,y,z1,c = 'r')
ax.scatter(x,y,z2,c = 'b')
ax.set_zlabel('body fat')
ax.set_xlabel('Weight lbs')
ax.set_ylabel('Heights inch')
plt.show
#scatter plot in 3D,and red points for estimated output,blue points for real output


temp_error_fitness_value = eq2(my_y_hat,test_output)
error = temp_error_fitness_value/100 - 1
print(error)
#find out the overall error for testing dataset

