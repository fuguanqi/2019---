#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:06:11 2019

@author: wxq
"""

import numpy as np
import xlrd
import math

workbook = xlrd.open_workbook('/Users/wxq/Documents/2019年中国研究生数学建模竞赛F题/附件1：数据集1-终稿.xlsx')
sheet2 = workbook.sheet_by_name('data1')
a = np.empty((sheet2.nrows-2,sheet2.ncols-1))

for i in range(2,sheet2.nrows):
    for j in range(sheet2.ncols-1):
        a[i-2][j] = sheet2.cell(i,j).value
        
#print(a[-1])
pos_sort_by_x = a[a[:, 1].argsort()] # 按第2列进行排序
#print(pos_sort_by_x[0][1])

def cal_total_distance(a,b,c):
    long_distance = 0
    for i in range(len(a)-1):
        d1 = a[i+1]-a[i]
        d2 = b[i+1]-b[i]
        d3 = c[i+1]-c[i]
        long_distance += math.sqrt(d1**2+d2**2+d3**2)
    return long_distance
    
def cal_dis(t1, t2):
    d1 = t2[0]-t1[0]
    d2 = t2[1]-t1[1]
    d3 = t2[2]-t1[2]
    dis = math.sqrt(d1**2+d2**2+d3**2)
    return dis

def greedy_generate(arr):
    temp_count = 0
    ALPHA1 = 25
    ALPHA2 = 15
    BETA1 = 20
    BETA2 = 25
    THETA = 30
    DELTA = 0.001
    result = []
    cur_point = 0
    cur_error_v = 0
    cur_error_h = 0
    flag_li = []
    flag = 0
    Next = 0
    while arr[cur_point][1] < arr[-1][1] and cur_point < len(arr)-1:
        
        #print(arr[cur_point][1])
        
        t = cur_point + 1
        max_dx = min((ALPHA1-cur_error_v)/DELTA, \
                     (ALPHA2-cur_error_h)/DELTA, \
                     (BETA1-cur_error_v)/DELTA, \
                     (BETA2-cur_error_h)/DELTA)
        #print(max_dx)
        #print('1',(ALPHA1-cur_error_v)/DELTA,'2',(ALPHA2-cur_error_h)/DELTA,'3',(BETA1-cur_error_v)/DELTA,'4',(BETA2-cur_error_h)/DELTA)
        
        max_dis_to_B = min((THETA-cur_error_v)/DELTA,(THETA-cur_error_h)/DELTA)
        if max_dis_to_B>cal_dis((arr[-1][1],arr[-1][2],arr[-1][3]),(arr[cur_point][1],arr[cur_point][2],arr[cur_point][3])):
            break
        
        if max_dx==(ALPHA1-cur_error_v)/DELTA or max_dx==(BETA1-cur_error_v)/DELTA:
            
            #temp_count += 1
           
            while arr[t][1] < arr[cur_point][1]+max_dx and t < len(arr)-1:
                distance = cal_dis((arr[cur_point][1],arr[cur_point][2],arr[cur_point][3]),(arr[t][1],arr[t][2],arr[t][3]))
                if distance < max_dx and arr[t][4] == 1:
                    
                    Next = t
                    print('V_inner:', Next)
                t += 1
            print('V:', Next)
            result.append(Next)
            flag = 1
            flag_li.append(flag)
            cur_error_v = 0
            cur_error_h = cur_error_h+DELTA*cal_dis((arr[cur_point][1],arr[cur_point][2],arr[cur_point][3]),(arr[Next][1],arr[Next][2],arr[Next][3]))
        
        else:
            while arr[t][1] < arr[cur_point][1]+max_dx and t < len(arr)-1:
                #print('yes')
                distance = cal_dis((arr[cur_point][1],arr[cur_point][2],arr[cur_point][3]),(arr[t][1],arr[t][2],arr[t][3]))
                #print(distance)
                if distance < max_dx and arr[t][4] == 0:
                    
                    #print('aaa')
                    
                    Next = t
                    print('H_inner:', Next)
                t += 1
            #print('bbb')
            #print(Next)
            print('H:', Next)
            result.append(Next)
            flag = 2
            flag_li.append(flag)
            cur_error_h = 0
            cur_error_v = cur_error_v+DELTA*cal_dis((arr[cur_point][1],arr[cur_point][2],arr[cur_point][3]),(arr[Next][1],arr[Next][2],arr[Next][3]))
        
        cur_point = Next
    return result, flag_li
    #print(temp_count)
#greedy_generate(pos_sort_by_x)
ret, ca_type= greedy_generate(pos_sort_by_x)
print('ret:', ret)
calibration_point = list(set(ret))
last_result = sorted(calibration_point)
print('last_result:', last_result)
#print(last_result)
#print(pos_sort_by_x[1][4])
#print(len(last_result))
#fd = open('')
X = []
Y = []
Z = []
A_point = [0.00,50000.00,5000.00]
B_point = [100000.00,59652.3433795158,5022.00116448164]
for i in last_result:
    X.append(round(pos_sort_by_x[i][1],2))
    Y.append(round(pos_sort_by_x[i][2],2))
    Z.append(round(pos_sort_by_x[i][3],2))
    #print(round(pos_sort_by_x[i][3],2))
    #print(round(pos_sort_by_x[i][2],2))
    #print(round(pos_sort_by_x[i][3],2))
X.append(B_point[0])
Y.append(B_point[1])
Z.append(B_point[2])
X.insert(0,A_point[0])
Y.insert(0,A_point[1])
Z.insert(0,A_point[2])
#print('X:',X)
#print('Y:',Y)
#print('Z:',Z)
final_dis = cal_total_distance(X,Y,Z)
print(final_dis)
print('flag_li:', ca_type)
print(pos_sort_by_x[80][4])