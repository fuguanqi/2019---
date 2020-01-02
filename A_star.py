import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math


Origin_data = pd.read_excel('../data/attach1.xlsx')                       #原始数据

PointA_data = Origin_data.loc[Origin_data['type'] == 'A']                  # A点数据
PointB_data = Origin_data.loc[Origin_data['type'] == 'B']                  # B点数据

Point_other_data = Origin_data.loc[(Origin_data['type'] != 'A') & (Origin_data['type'] != 'B')]    # 除A，B两点之外的数据

Vertical_data = Point_other_data.loc[Point_other_data['flag'] == 0]       # 水平校正点
Horizontal_data = Point_other_data.loc[Point_other_data['flag'] == 1]     # 垂直校正点

ALPHA1 = 25.0
ALPHA2 = 15.0
BETA1 = 20.0
BETA2 = 25.0
THETA = 30.0
DELTA = 0.001

Current_path = pd.DataFrame(columns=['num', 'X', 'Y', 'Z', 'type', 'flag'])  # 初始化搜寻路径

# 返回当前可向前搜寻的最大距离
def search_max(cur_error_v, cur_error_h):
    return min((ALPHA1-cur_error_v)/DELTA, (ALPHA2-cur_error_h)/DELTA,    # 越小越紧迫
               (BETA1-cur_error_v)/DELTA, (BETA2-cur_error_h)/DELTA)

# def greedy_generate():
#     # 从A 向 B 搜寻 , 向X轴增大的方向搜寻, 故初始时, 对候选点数按照x轴排序
#     point_data = Point_other_data.sort_values(by='X', ascending=True)
#     print(point_data)
#     # print(len(point_data))
#     cur_point=0    # 搜寻的index , 初始为第一个点
#
#     cur_error_v = 0.0   # 初始误差
#     cur_error_h = 0.0
#
#     print (point_data.iloc[cur_point, 1])
#     while point_data.iloc[cur_point,1] < 100000:
#         t = cur_point
#         next = 0
#         max_dx = search_max(cur_error_v, cur_error_h)
#         if max_dx == (ALPHA1-cur_error_v)/DELTA or max_dx == (BETA1-cur_error_v)/DELTA:
#             while point_data.iloc[t, 1] < point_data.iloc[cur_point, 1]:
#                 if cal_dis()

# 代价

def sort_by_f_star(path, neighbours):
    neighbours['f_star'] = 0
    # print(neighbours)
    for i in range(neighbours.shape[0]):
        neighbours.iloc[i, 6] = f_star(path, neighbours.iloc[i, :])

    neighbours = neighbours.sort_values(by='f_star', ascending=False)
    neighbours = neighbours.drop(columns=['f_star'])
    return neighbours

def f_star(path, point):
    # print (path)
    # print (point)
    return get_path_dis(path, path.shape[0]-1) + cal_dis(point, path.iloc[path.shape[0]-1, :]) + cal_dis(PointB_data.iloc[0, :], point)

def get_path_dis(path, index):
    i = 0
    result = 0
    while i < index:
        result += cal_dis(path.iloc[i, :], path.iloc[index, :])
    return result
    # if index < 1:
    #     return 0
    #
    # i = 2
    # cur_error_v = 0
    # cur_error_h = 0
    #
    # if index == 1:
    #     cur_error_v = DELTA * cal_dis(path.iloc[0, :], path.iloc[1, :])
    #     cur_error_h = DELTA * cal_dis(path.iloc[0, :], path.iloc[1, :])
    #
    # while i <= index:
    #     if path.iloc[i-1, 4] == 1:
    #         cur_error_v = 0 + DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])
    #         cur_error_h += DELTA * cal_dis(path.iloc[i - 1, :], path.iloc[i, :])
    #     else:
    #         cur_error_v += DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])
    #         cur_error_h = 0 + DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])
    #
    # return cur_error_v, cur_error_h

def get_error(path, index):
    if index < 1:
        return 0, 0

    i = 2
    cur_error_v = 0
    cur_error_h = 0

    if index == 1:
        cur_error_v = DELTA * cal_dis(path.iloc[0, :], path.iloc[1, :])
        cur_error_h = DELTA * cal_dis(path.iloc[0, :], path.iloc[1, :])

    while i <= index:
        if path.iloc[i-1, 4] == 1:
            cur_error_v = 0 + DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])
            cur_error_h += DELTA * cal_dis(path.iloc[i - 1, :], path.iloc[i, :])
        else:
            cur_error_v += DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])
            cur_error_h = 0 + DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])

    return cur_error_v, cur_error_h


def find_neighbours(point, path):
    neighbours = pd.DataFrame(columns=['num', 'X', 'Y', 'Z', 'type', 'flag'])    # 搜寻当前可达点
    # print (point, path)

    cur_error_v, cur_error_h = get_error(path, path.shape[0]-1)

    # print (point[4])
    if point[4] == 1:
        cur_error_v = 0
    else:
        cur_error_h = 0

    max_dx_v = min((ALPHA1-cur_error_v)/DELTA, (BETA1-cur_error_v)/DELTA)
    max_dx_h = min((ALPHA2-cur_error_h)/DELTA, (BETA2-cur_error_h)/DELTA)

    # print(max_dx_v, max_dx_h)

    max_dis_to_b = min((THETA - cur_error_v) / DELTA, (THETA - cur_error_h) / DELTA)

    # print (PointB_data[0, :])
    if max_dis_to_b >= cal_dis(PointB_data.iloc[0, :], point):
        return PointB_data

    # print (type(point))
    temp = Origin_data[(Origin_data['X'] == point['X']) & (Origin_data['Y'] == point['Y']) & (Origin_data['Z'] == point['Z'])]
    point_index = temp.index.tolist()[0]

    t = point_index + 1
    # print (cal_dis(Origin_data.iloc[point_index, :], Origin_data.iloc[t, :]))
    print (max_dx_v, max_dx_h)

    while (Origin_data.iloc[t, 1] - Origin_data.iloc[point_index, 1]) < max_dx_v:
        # print(Origin_data.iloc[t, 4])
        if Origin_data.iloc[t, 4] == 1:
            neighbours = neighbours.append(Origin_data.iloc[t, :])
        t += 1

    t = point_index + 1
    while (Origin_data.iloc[t, 1] - Origin_data.iloc[point_index, 1]) < max_dx_h:
        if Origin_data.iloc[t, 4] == 0:
            neighbours = neighbours.append(Origin_data.iloc[t, :])
        t += 1

    # print (neighbours)
    return neighbours

def a_star_path(point, current_path):
    print (a_star_path)
    current_path.loc[current_path.shape[0]+1] = list(point)

    neighbours = find_neighbours(point, current_path)

    if neighbours.iloc[0, 4] == 'B':
        current_path.loc[current_path.shape[0] + 1] = PointB_data.iloc[0, :]
        return current_path
    neighbours = sort_by_f_star(current_path, neighbours)

    i = neighbours.shape[0] - 1
    while i >= 0:
        print('i:', i)
        ret = a_star_path(neighbours.iloc[i, :], current_path)
        if ret != False:
            return ret
        i -= 1
    return False

# 初始数据处理, 将点数按照 x的位置生序排列, 执行find_neighbours() 时前向搜索
# def inital_data():
#     Origin_data.sort_values(by='X', ascending=True)
#     print (Origin_data)
#     # print (Point_other_data)

# 空间两点计算
def cal_dis(point1, point2):
    d_x = point1['X'] - point2['X']
    d_y = point1['Y'] - point2['Y']
    d_z = point1['Z'] - point2['Z']
    return math.sqrt(d_x**2 + d_y**2 + d_z**2)


if __name__ == '__main__':
    current_path = pd.DataFrame(columns=['num', 'X', 'Y', 'Z', 'type', 'flag'])  # 初始化搜寻路径
    # print (Origin_data)

    a_star_path(PointA_data.iloc[0, :], current_path)
    # a_star_path(list(PointB_data.iloc[0, :]), current_path)
    # print(current_path)

    # greedy_generate()
    # print(Origin_data)