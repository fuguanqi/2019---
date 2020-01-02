# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import matplotlib as mpl
import random
import math
from mpl_toolkits.mplot3d import Axes3D

# 模拟退火算法框架
# def algorithm():
#     # 初始解的生成，（目前的做法是贪心算法求出最优解，若求出8个点，则8个点的组合可能有多种情况，分别作为不同的种子来计算）
#     construct init x_0 and x_current = x_0
#     set initial temperature T = T_0       # 初始温度生成
#     while continuing criterion do{
#         for i = 1 to T_L do{              # 等温步数的选取，通常要选择比较大的， 温度高时比较小，温度低时比较大，可以写作一个温度的函数，但一般设置比较大就ok
#             generate randomly a neighbouring solution x_p belong N(x_current)   # 变异函数
#             compute cost = C(x_p) - C(x_current)
#             if cost <= 0 or random(0, 1) < exp(-cost/(kT)):
#                 x_current = x_p  #接受当前的解
#             endif
#         }
#         set new temperature T = decrease(T)       # 降温方式的选取 1.经典 T(t) = T0/log(1+t)  2.快速 T(t) = T0/(1+t)  3 T(t+det) = pT(t) (0.8 <= p <= 0.99)
#     }
#     return solution    # 找到最小代价函数的

#--------------------------------- 全局变量处理------------------------------

# 第一组数据集
ALPHA1 = 25.0
ALPHA2 = 15.0
BETA1 = 20.0
BETA2 = 25.0
THETA = 30.0
DELTA = 0.001

# # 第二组数据集
# ALPHA1 = 20.0
# ALPHA2 = 10.0
# BETA1 = 15.0
# BETA2 = 20.0
# THETA = 20.0
# DELTA = 0.001

Origin_data = pd.read_excel('../data/attach1.xlsx')                       #原始数据

PointA_data = Origin_data.loc[Origin_data['type'] == 'A']                  # A点数据
PointB_data = Origin_data.loc[Origin_data['type'] == 'B']                  # B点数据

Point_other_data = Origin_data.loc[(Origin_data['type'] != 'A') & (Origin_data['type'] != 'B')]    # 除A，B两点之外的数据

Vertical_data = Point_other_data.loc[Point_other_data['flag'] == 1]       # 垂平校正点
Horizontal_data = Point_other_data.loc[Point_other_data['flag'] == 0]     # 水平直校正点

DIS_MEAN = 0
DIS_STD = 0
COUNT_MEAN = 0
COUNT_STD = 0

#---------------------------------- 模拟退火算法相关----------------------------
# 空间两点计算
def cal_dis(point1, point2):
    d_x = point1['X'] - point2['X']
    d_y = point1['Y'] - point2['Y']
    d_z = point1['Z'] - point2['Z']
    return math.sqrt(d_x**2 + d_y**2 + d_z**2)

# 状态产生函数, 模拟退火中的畸变函数
def generate_new(path):
    while True:
        alt_point_index = random.randint(1, path.shape[0]-2)                        # 随机选择变异点的下标
        new_path = path.iloc[0:alt_point_index, :]
        point = path.iloc[alt_point_index, :]

        # print('****alt_point_index****',alt_point_index)
        # print('*******new_path before *****')
        # print(new_path)

        neighbours = find_neighbours(path.iloc[alt_point_index-1, :], new_path)
        # print(point)
        # print(new_path)
        # print('*******neighbours******')
        # print(neighbours)

        temp = neighbours[(neighbours['X'] == point['X']) & (neighbours['Y'] == point['Y']) & (neighbours['Z'] == point['Z'])]
        # point_index = temp.index.tolist()
        # print(point_index)
        # print('***temp index****',temp.index)
        neighbours.drop(temp.index, inplace=True)
        # print('*******neighbours after delete*****')
        # print(neighbours)

        neighbours = sort_by_f_star(path, neighbours)

        i = neighbours.shape[0] - 1
        while i >= 0:
            # print('***i****', i)
            ret = a_star_path(neighbours.iloc[i, :], new_path)
            if isinstance(ret, bool):
                i -= 1
            else:
                # new_path = pd.concat([new_path, ret], ignore_index=True)
                new_path = new_path.append(ret)
                # print('**** new path after*****')
                # print(new_path)
                return new_path


# 去量纲操作
def cal_stat(path):
    dis = []                # 用于记录,距离
    count = []              # 用于记录,校正次数

    for i in range(30):
        # print('decode lianggang', i)
        new_path = generate_new(path)
        dis.append(get_path_dis(new_path, new_path.shape[0]-1))
        count.append(new_path.shape[0])

    print('***dis:***', dis)
    print('****count****',count)

    global DIS_MEAN
    DIS_MEAN = np.mean(dis)

    global DIS_STD
    DIS_STD = np.var(dis)

    global COUNT_MEAN
    COUNT_MEAN = np.mean(count)

    global COUNT_STD
    COUNT_STD = np.std(count)

    print('decode lianggang result',DIS_MEAN, DIS_STD, COUNT_MEAN, COUNT_STD)

def cal_cost(path):
    w1 = 1
    w2 = 1
    dis = get_path_dis(path, path.shape[0]-1)
    count = path.shape[0]
    return w1*(dis-DIS_MEAN)/DIS_STD + w2*(count-COUNT_MEAN)/COUNT_STD

def sort_by_f_star(path, neighbours):
    neighbours['f_star'] = 0
    # print(neighbours)
    for i in range(neighbours.shape[0]):
        neighbours.iloc[i, 6] = f_star(path, neighbours.iloc[i, :])

    neighbours = neighbours.sort_values(by='f_star', ascending=False)
    neighbours = neighbours.drop(columns=['f_star'])
    return neighbours

def f_star(path, point):
    return cal_dis(PointB_data.iloc[0, :], point)
    # return get_path_dis(path, path.shape[0]-1) + cal_dis(point, path.iloc[path.shape[0]-1, :]) + cal_dis(PointB_data.iloc[0, :], point)

def get_path_dis(path, index):
    i = 0
    result = 0
    while i < index:
        result += cal_dis(path.iloc[i, :], path.iloc[i+1, :])
        # print(result)
        i += 1
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
        # print('cur_error_v_before', 0)
        # print('cur_error_h_before', 0)
        return 0, 0

    cur_error_v = 0
    cur_error_h = 0

    if index == 1:
        cur_error_v = DELTA * cal_dis(path.iloc[0, :], path.iloc[1, :])
        cur_error_h = DELTA * cal_dis(path.iloc[0, :], path.iloc[1, :])

        # print('cur_error_v_before', cur_error_v)
        # print('cur_error_h_before', cur_error_h)
        return cur_error_v, cur_error_h

    i = 1
    while i <= index:
        if path.iloc[i-1, 4] == 'A':
            cur_error_v = DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])
            cur_error_h = DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])

        elif path.iloc[i-1, 4] == 1:
            cur_error_v = 0 + DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])
            cur_error_h += DELTA * cal_dis(path.iloc[i - 1, :], path.iloc[i, :])

        else:
            cur_error_v += DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])
            cur_error_h = 0 + DELTA * cal_dis(path.iloc[i-1, :], path.iloc[i, :])

        i += 1
    # print('cur_error_v_before', cur_error_v)
    # print('cur_error_h_before', cur_error_h)
    return cur_error_v, cur_error_h


def find_neighbours(point, path):
    neighbours = pd.DataFrame(columns=['num', 'X', 'Y', 'Z', 'type', 'flag'])    # 搜寻当前可达点
    # print (point, path)

    cur_error_v, cur_error_h = get_error(path, path.shape[0]-1)        # 不包含 point

    # if path.iloc[path.shape[0]-1, 4] == 1:
    #     cur_error_v = 0
    # elif path.iloc[path.shape[0]-1, 4] == 0:
    #     cur_error_h = 0
    # else:
    #     cur_error_h = 0
    #     cur_error_v = 0

    if point[4] == 1:
        cur_error_v = 0
        # cur_error_h += DELTA * cal_dis(point, path.iloc[path.shape[0]-2, :])    # point调整后误差

    elif point[4] == 0:
        cur_error_h = 0
        # cur_error_v += DELTA * cal_dis(point, path.iloc[path.shape[0]-2, :])

    elif point[4] == 'A':
        cur_error_v = 0
        cur_error_h = 0

    # elif point[4] == 'B':
    #     cur_error_h += DELTA * cal_dis(point, path.iloc[path.shape[0]-2, :])
    #     cur_error_v += DELTA * cal_dis(point, path.iloc[path.shape[0]-2, :])

    # print('cur_error_v_after', cur_error_v)
    # print('cur_error_h_after', cur_error_h)

    # max_dx_v = min((ALPHA1-cur_error_v)/DELTA, (BETA1-cur_error_v)/DELTA)
    # max_dx_h = min((ALPHA2-cur_error_h)/DELTA, (BETA2-cur_error_h)/DELTA)

    max_dx_a1 = (ALPHA1-cur_error_v)/DELTA
    max_dx_a2 = (ALPHA2-cur_error_h)/DELTA
    max_dx_b1 = (BETA1-cur_error_v)/DELTA
    max_dx_b2 = (BETA2-cur_error_h)/DELTA

    max_dx = max(max_dx_a1, max_dx_a2, max_dx_b1, max_dx_b2)
    min_dx = min(max_dx_a1, max_dx_a2, max_dx_b1, max_dx_b2)

    # max_dx_v = min((ALPHA1-cur_error_v)/DELTA, (BETA1-cur_error_v)/DELTA)
    # max_dx_h = min((ALPHA2-cur_error_h)/DELTA, (BETA2-cur_error_h)/DELTA)

    max_dis_to_b = min((THETA - cur_error_v) / DELTA, (THETA - cur_error_h) / DELTA)
    if max_dis_to_b >= cal_dis(PointB_data.iloc[0, :], point):
        return PointB_data

    temp = Origin_data[(Origin_data['X'] == point['X']) & (Origin_data['Y'] == point['Y']) & (Origin_data['Z'] == point['Z'])]
    point_index = temp.index.tolist()[0]

    # print('point index', point_index)
    t = point_index + 1
    # print (cal_dis(Origin_data.iloc[point_index, :], Origin_data.iloc[t, :]))

    # print('max dx v', max_dx_v)
    # print('max dx h', max_dx_h)

    while t < Origin_data.shape[0] and math.fabs(Origin_data.iloc[t, 1] - Origin_data.iloc[point_index, 1]) < min_dx:
        dis = cal_dis(Origin_data.iloc[t, :], Origin_data.iloc[point_index, :])
        if dis <= min_dx:
        # if (dis < max_dx_a1 and dis < max_dx_a2 and cur_error_v+dis*DELTA < BETA1 and cur_error_h+dis*DELTA < BETA2 and  Origin_data.iloc[t, 4] == 1) \
        #         or (dis < max_dx_b1 and dis < max_dx_b2 and Origin_data.iloc[t, 4] == 0):
        # if cal_dis(Origin_data.iloc[t, :] , Origin_data.iloc[point_index, :]) < max_dx:
            check_point = path[(Origin_data.iloc[t, 1] == path['X']) & (Origin_data.iloc[t, 2] == path['Y']) & (Origin_data.iloc[t, 3]== path['Z'])]
            if check_point.empty:
                neighbours = neighbours.append(Origin_data.iloc[t, :])
        t += 1

    # t = point_index - 1
    # print (cal_dis(Origin_data.iloc[point_index, :], Origin_data.iloc[t, :]))

    # print('max dx v', max_dx_v)
    # print('max dx h', max_dx_h)

    # while t >0 and math.fabs(Origin_data.iloc[t, 1] - Origin_data.iloc[point_index, 1]) < min_dx:
    #     dis = cal_dis(Origin_data.iloc[t, :], Origin_data.iloc[point_index, :])
    #     if dis <= min_dx:
    #         # if (dis < max_dx_a1 and dis < max_dx_a2 and cur_error_v+dis*DELTA < BETA1 and cur_error_h+dis*DELTA < BETA2 and  Origin_data.iloc[t, 4] == 1) \
    #         #         or (dis < max_dx_b1 and dis < max_dx_b2 and Origin_data.iloc[t, 4] == 0):
    #         # if cal_dis(Origin_data.iloc[t, :] , Origin_data.iloc[point_index, :]) < max_dx:
    #         check_point = path[(Origin_data.iloc[t, 1] == path['X']) & (Origin_data.iloc[t, 2] == path['Y']) & (Origin_data.iloc[t, 3]== path['Z'])]
    #         if check_point.empty:
    #             neighbours = neighbours.append(Origin_data.iloc[t, :])
    #     t -= 1

    # while t < Origin_data.shape[0] and (Origin_data.iloc[t, 1] - Origin_data.iloc[point_index, 1]) < max_dx_v:
    #     if cal_dis(Origin_data.iloc[t, :] , Origin_data.iloc[point_index, :]) < max_dx_v and Origin_data.iloc[t, 4] == 1:
    #         neighbours = neighbours.append(Origin_data.iloc[t, :])
    #     t += 1
    #
    # t = point_index + 1
    # while t < Origin_data.shape[0] and (Origin_data.iloc[t, 1] - Origin_data.iloc[point_index, 1]) < max_dx_h:
    #     if cal_dis(Origin_data.iloc[t, :], Origin_data.iloc[point_index, :]) < max_dx_h and Origin_data.iloc[t, 4] == 0:
    #         neighbours = neighbours.append(Origin_data.iloc[t, :])
    #     t += 1
    return neighbours

def a_star_path(point, path):
    # path.loc[path.shape[0]] = list(point)
    path = path.append(point)
    # path = path.concat([path, point], ignore_index=False)
    # print ('***********************current path********************')
    # print(path)

    neighbours = find_neighbours(point, path)

    if neighbours.shape[0] == 0:
        return False

    if neighbours.iloc[0, 4] == 'B':
        # path.loc[path.shape[0] + 1] = PointB_data.iloc[0, :]
        path = path.append(PointB_data.iloc[0, :])
        return path

    # if neighbours['type'].sum() == neighbours.shape[0] or neighbours['type'].sum() == 0:
    #     return False

    neighbours = sort_by_f_star(path, neighbours)
    # print('-----------------------neighbours-------------------------')
    # print(neighbours)
    # print('neighbours', neighbours['type'].sum())

    i = neighbours.shape[0] - 1
    while i >= 0:
        ret = a_star_path(neighbours.iloc[i, :], path)
        if isinstance(ret, bool):
            i -= 1
        else:
            return ret
    return False


# 模拟退火算法
def algorithm_sa():

    input_num = [0, 200, 354, 294, 91, 28, 183, 581, 194, 288, 561, 53, 3, 612]
    input_x = [0.0, 12142.2205868705, 13255.4105755552, 21191.7729307933, 28387.1824995451, 32952.0789549698, 44139.6856850401, 46532.8450612656, 57582.3792740119, 58196.7743465736, 66726.2371168726,75079.1105406502, 77991.5459109057, 100000.0]
    input_y = [50000.0, 56740.4232656757, 58008.2135629962, 57198.4754529258, 58660.0055141931, 60824.8649038331, 62558.049781309, 64957.2424132102, 64320.5720267624, 67621.5803410544, 64030.6571657957, 65228.2494101141, 63982.1752857149, 59652.3433795158]
    input_z = [5000.0, 3733.39365592965, 2275.16405898938, 6851.28419162468, 7277.4807331403, 2263.99124399827, 2660.04247005842, 2084.10369039046, 3868.1250444089, 5278.05561374484, 8825.61539613089,9210.79020906868, 5945.82303761753, 5022.00116448164]
    input_type = ['A', 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 'B']
    input_flag = [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]

    # input_num = [0, 521, 40, 200, 503, 69, 80, 91, 607, 61, 278, 369, 250, 172, 340, 583, 612]
    # input_x = [0, 9522, 9019, 12142, 11392, 19835, 27810, 28387, 36706,44838, 50065, 61296, 60551, 68965, 73028, 80222, 100000]
    # input_y = [50000, 50387, 50399, 56740, 56973, 59347, 57543, 58660, 58334, 58254,56062, 54653, 55058, 56298, 52430, 56403, 59652]
    # input_z = [5000, 3644, 2569, 3733, 4097, 4903, 5123, 7277, 7953, 7374, 5606,7234, 9070, 7240, 9758, 7760, 5022]
    # input_type = ['A', 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 'B']
    # input_flag = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]
    input = {'num':input_num, 'X':input_x, 'Y':input_y, 'Z':input_z, 'type':input_type, 'flag':input_flag}

    solution = pd.DataFrame(input)
    # print('initial solution')
    # print(solution)
    # print('initial total distance', get_path_dis(solution, solution.shape[0]-1))

    # cal_stat(solution)  # 去量纲

    temperature = 100.0
    cooling_rate = 0.8
    iterations = 1               # 迭代步长, 用于控制降温循环
    # cost = cal_cost(solution)
    # print('initial cost', cost)

    # while temperature > 1.0:
    #     print('temperature', temperature)
    #     temp_solution = generate_new(solution)
    #     current_cost = cal_cost(temp_solution)
    #     print('current cost', current_cost)
    #     diff = current_cost - cost
    #
    #     rand = random.random()
    #     print('diff, rand', diff, rand)
    #     if diff < 0: # or rand < math.exp(-diff/(temperature*0.1)):
    #         solution = temp_solution
    #         cost = current_cost
    #         iterations = iterations + 1
    #
    #     if iterations >= 10:         #每10步降温
    #         temperature = cooling_rate*temperature
    #         iterations = 1

    solution = generate_new(solution)
    return solution
#------------------------------------数据读写处理-------------------------------
# 问题 结果写excel
def write_data():
    pass

# 模拟退火算法结果绘制
def draw(current_path):
    x_v = list(Vertical_data.iloc[:, 1])
    y_v = list(Vertical_data.iloc[:, 2])
    z_v = list(Vertical_data.iloc[:, 3])

    x_h = list(Horizontal_data.iloc[:, 1])
    y_h = list(Horizontal_data.iloc[:, 2])
    z_h = list(Horizontal_data.iloc[:, 3])

    x_a = list(PointA_data.iloc[:, 1])
    y_a = list(PointA_data.iloc[:, 2])
    z_a = list(PointA_data.iloc[:, 3])

    x_b = list(PointB_data.iloc[:, 1])
    y_b = list(PointB_data.iloc[:, 2])
    z_b = list(PointB_data.iloc[:, 3])

    pic = plt.figure(figsize=(12, 8), dpi=100).add_subplot(projection='3d')

    # 加上测试点, 拉伸z轴范围
    test_x = [0]
    test_y = [0]
    test_z = [100000]

    # 绘制点
    pic.scatter(x_a, y_a, z_a, c='000000')
    pic.text(x_a[0], y_a[0], z_a[0], 'Point A', fontsize=20)

    pic.scatter(x_b, y_b, z_b, c='000000')
    pic.text(x_b[0], y_b[0], z_b[0], 'Point B', fontsize=20)

    pic.scatter(x_v, y_v, z_v, c='orange', label='Point Vertical')
    pic.scatter(x_h, y_h, z_h, c='b', label='Point Horizontal')
    # pic.scatter(test_x, test_y, test_z, c='w', alpha=0)  # 暂时未找到 拉伸z轴的方法, 故多绘制一个透明度为0的白色点, 让其z坐标为100000

    re_a_star_x = list(current_path['X'])
    re_a_star_y = list(current_path['Y'])
    re_a_star_z = list(current_path['Z'])

    # 绘制折线
    pic.plot(re_a_star_x,re_a_star_y ,re_a_star_z, label='path')

    # 辅助图示
    pic.legend(loc='upper left', fontsize=12)
    plt.title('Aircraft path', fontsize=15)
    pic.set_xlabel('X')
    pic.set_ylabel('Y')
    pic.set_zlabel('Z')

    plt.draw()
    plt.show()

if __name__ == '__main__':
    # current_path = pd.DataFrame(columns=['num', 'X', 'Y', 'Z', 'type', 'flag'])      # 初始化搜寻路径
    solution = algorithm_sa()

    print('******final solution*******')
    print(solution)
    print('final total distance', get_path_dis(solution, solution.shape[0]-1))

    i = 0
    while i < solution.shape[0]:
        cur_error_v, cur_error_h = get_error(solution, i)
        print(i, cur_error_v, cur_error_h)
        i += 1

    draw(solution)

    # current_path = pd.concat([PointA_data, PointB_data], ignore_index=True)

    # read_data()
