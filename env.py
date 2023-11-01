import random

import numpy as np

def action_space_f():
    '''
    :return: (112 * 9)
    '''
    action_space_one = []
    for action_space_sb in block_size:
        for action_space_k in block_num:
            for action_spcae_w in range(7):
                a = [1,1,1,1,1,1,1,0,0]
                a[action_spcae_w] = 2
                a[7] = action_space_sb
                a[8] = action_space_k
                action_space_one.append(a)
    return np.array(action_space_one)



# 用户端数量 UE

num_users = 3
# 基站端数量 BS
num_base_stations = 4
# 信噪比离散数据集合 SNR
snr_values = [1, 7, 15]
# 计算资源集合  Hz
compute_resources = [200, 500, 1000]

state_trans_p_m = np.array([[0.6, 0.3, 0.1],\
                             [0.3, 0.1, 0.6],\
                             [0.1, 0.6, 0.3]])
compute_trans_p_m = np.array([[0.7, 0.2, 0.1],\
                             [0.2, 0.1, 0.7],\
                             [0.1, 0.7, 0.2]])
action_space = np.zeros(9)
states = np.zeros(14)

g = 1.5

block_size = [0.5, 1, 2, 4]  # Mb
block_num = [2, 4, 6, 8]
Band_width = 20  #MHz

weight_blo_sys = 0.5
weight_mec_sys = 0.5

cpu_sig = 0.001     # G cycles b
cpu_MAC = 0.01      # G cycles 0
cpu_smartc = 0.01  # G cycles  a
#average_size_trans = 100  # B
average_size_trans = 0.0001  # Mb
T_dot = 2

current_p_node_index = 1

a_actions = action_space_f()
'''                    0
    states = np.array([pn, \
     1         2         3       4        5        6      |  7         8         9    
    b1_2_snr, b1_3snr, b1_4snr, b2_3snr, b2_4snr, b3_4snr, c1_psnr, c2_psnr, c3_psnr, \
    10     11     12      13
    b1_cr, b2_cr, b3_cr, b4_cr])
    '''
'''
0-4  4-7 7 8
    [Wu1, Wu2, Wu3, Wb1, Wb2, Wb3, Wb4, Sb ,K]
'''
# def init_state():
#     ''' example
#           b1    b2    b3   b4     c1    c2     c3    pn   cr
#     b1[[   0.    7.    7.   15.    1.   15.    7.    0.  500.]
#     b2 [  15.    0.    7.   15.    1.    7.    7.    0. 1000.]
#     b3 [  15.    1.    0.   15.    1.   15.    1.    1.  200.]
#     b4 [   1.    7.    1.    0.    1.    1.    7.    0.  500.]]'''
#     states = np.zeros([4,9])
#     for bu in range(num_base_stations+num_users):
#         for b in range(num_base_stations):
#             states[b][bu] = int(snr_values[random.randint(0, 2)]) if b != bu else 0
#     states[random.randint(0, 3)][7] = 1
#     for i in range(num_base_stations):
#         states[i][8] = int(compute_resources[random.randint(0, 2)])
#     return states