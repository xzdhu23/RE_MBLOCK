import math
import random
import numpy as np
from env import block_size, states, current_p_node_index, snr_values, compute_resources, block_num, a_actions, \
    average_size_trans, Band_width, cpu_sig, g, cpu_MAC, cpu_smartc, T_dot, state_trans_p_m, compute_trans_p_m, \
    weight_blo_sys, weight_mec_sys


def init_state():
    '''                  0
    states = np.array([pn, \
     1         2         3       4        5        6      |  7         8         9
    b1_2_snr, b1_3snr, b1_4snr, b2_3snr, b2_4snr, b3_4snr, c1_psnr, c2_psnr, c3_psnr, \
    10     11     12      13
    b1_cr, b2_cr, b3_cr, b4_cr])
    '''

    states[0] = current_p_node_index
    for i in range(1, 10):
        states[i] = random.choice(snr_values)
    for j in range(10, 14):
        states[j] = random.choice(compute_resources)

    return states


def trans_rate(w1, w0, snr):  # 公式（2）（3）
    w = w0 / 8 * w1
    return w1 * w * math.log10(1 + snr)


def next_p_node(curretnode):
    if curretnode == 4:
        next_p_node_num = 1
    else:
        next_p_node_num = curretnode + 1
    return next_p_node_num


def select_p_p1(pr, ne):
    if pr == 1:
        if ne == 2:
            return 1
        elif ne == 3:
            return 2
        else:
            return 3
    elif pr == 2:
        if ne == 3:
            return 4
        elif ne == 4:
            return 5
    elif pr == 3:
        if ne == 4:
            return 6


def step(state, action):
    '''
    根据状态转移概率矩阵得到下一阶段的状态
    '''

    # 得到当前state的区块大小、区块个数和主节点索引
    action = a_actions[action]
    Sb = action[-2]
    K = action[-1]
    primary_node_index = int(state[0])
    # print("当前阶段state:{}".format(state))
    # print("当前阶段action:{}".format(action))

    # 为方便计算将state部分值从1维变为2维
    state_m = np.zeros([4, 4])
    flag = 1
    for ii in [1, 2, 3]:
        for ij in range(ii, 4):
            state_m[ij][ii - 1] = state[flag]
            flag += 1
    state_m = state_m.T + state_m
    # print(state_m)

# PBFT算法实现
    # 共识时间列表
    T_tr_req_list = []
    T_tr_prep_list = []
    T_tr_pre_list = []
    T_tr_comm_list = []
    T_tr_reply_list = []
    # 处理时间列表
    C_tr_req_list = []
    C_tr_prep_list = []
    C_tr_pre_list = []
    C_tr_comm_list = []
    C_tr_reply_list = []
# Request阶段
    for t_c_rate in range(0, 3):
        T_tr_req = average_size_trans / trans_rate(action[t_c_rate], Band_width, state[t_c_rate + 7])  # 公式（6）
        T_tr_req_list.append(T_tr_req.__round__(6))
        C_req = (Sb*(cpu_sig+cpu_MAC)/(g*average_size_trans)+(Sb*cpu_smartc/average_size_trans))/state[t_c_rate + 10]
        C_tr_req_list.append(C_req.__round__(6))
    max_req_t = max(T_tr_req_list)
    max_req_c = max(C_tr_req_list)
    # print("Request_tr:{}, c:{}".format(max_req_t, max_req_c))

# Pre-prepare阶段的传输时间
    for t_pb_rate in range(4):
        if primary_node_index - 1 == t_pb_rate:
            c_prep = (cpu_sig + 3*cpu_MAC)/state[t_pb_rate + 10]
            C_tr_prep_list.append(c_prep.__round__(6))
            continue
        else:
            T_pb_rate = Sb / trans_rate(action[primary_node_index + 3], Band_width, state_m[primary_node_index - 1][t_pb_rate])
            T_tr_prep_list.append(T_pb_rate.__round__(6))
            C_prep = (cpu_sig + cpu_MAC + ((cpu_sig + cpu_MAC + cpu_smartc)*Sb/average_size_trans))/state[t_pb_rate + 10]
            C_tr_prep_list.append(C_prep.__round__(6))
    max_prep_t = max(T_tr_prep_list)
    max_prep_c = max(C_tr_prep_list)
    # print("Pre-prepare_tr:{}, c:{}".format(max_prep_t, max_prep_c))

# Prepare阶段
    for t_prb_rate in range(4):
        if t_prb_rate + 1 == primary_node_index:
            C_tr_pre_list.append(3*(cpu_sig + cpu_MAC)/state[t_prb_rate + 10])
            continue
        else:
            for t_prb_rate_2 in range(4):
                if t_prb_rate_2 == t_prb_rate:
                    continue
                else:
                    T_prb_rate = Sb / trans_rate(action[primary_node_index + 3], Band_width, state_m[t_prb_rate][t_prb_rate_2])
                    C_prb = (3 * (cpu_sig + cpu_MAC) + cpu_sig + 3 * cpu_sig / state[t_prb_rate + 10])
                    T_tr_pre_list.append(T_prb_rate.__round__(6))
                    C_tr_pre_list.append(C_prb.__round__(6))
    max_pre_t = max(T_tr_pre_list)
    max_pre_c = max(C_tr_pre_list)
    # print("Prepare_tr:{} , c:{}".format(max_pre_t, max_pre_c))

# Commit阶段
    for t_comm_rate in range(4):
        for t_comm_rate_2 in range(4):
            if t_comm_rate_2 == t_comm_rate:
                continue
            else:
                T_comm_rate = Sb / trans_rate(action[primary_node_index + 3], Band_width, state_m[t_comm_rate][t_comm_rate_2])
                C_comm = (3 * (cpu_sig + cpu_MAC) + cpu_sig + 3 * cpu_sig / state[t_comm_rate_2 + 10])
                T_tr_comm_list.append(T_comm_rate.__round__(6))
                C_tr_comm_list.append(C_comm.__round__(6))
    max_comm_t = max(T_tr_comm_list)
    max_comm_c = max(C_tr_comm_list)
    # print("Commit_tr:{} ,{}".format(max_comm_t, max_comm_c))

# Reply阶段
    for t_reply_rate in range(4):
        if t_reply_rate == primary_node_index - 1:
            C_reply = (3 * (cpu_sig + cpu_MAC) / state[t_reply_rate + 10])
            C_tr_reply_list.append(C_reply)
            continue
        else:
            T_rep_rate = Sb / trans_rate(action[primary_node_index + 3], Band_width, state_m[t_reply_rate][primary_node_index - 1])
            C_reply = (Sb * (cpu_sig + cpu_MAC) / average_size_trans / state[t_reply_rate + 10])
            T_tr_reply_list.append(T_rep_rate.__round__(6))
            C_tr_reply_list.append(C_reply.__round__(6))
    max_reply_t = max(T_tr_comm_list)
    max_reply_c = max(C_tr_reply_list)
    # print("Reply_tr:{} ,{}".format(max_reply_t, max_reply_c))

    tao = 2
    T_p = min(max_req_t, tao) + min(max_prep_t, tao) + min(max_pre_t, tao) + min(max_comm_t, tao) + min(max_reply_t, tao)
    T_c = min(max_req_c, tao) + min(max_prep_c, tao) + min(max_pre_c, tao) + min(max_comm_c, tao) + min(max_reply_c, tao)
    T_p = T_p.__round__(6)
    T_c = T_c.__round__(6)
    T_f = (T_c + T_p).__round__(6)
    # print("PBFT总传输时间={},总处理时间={} ,  总时间={}".format(T_p, T_c, T_f))


    # 当前state下MEC性能评估
    ver_mac_smc_time = cpu_sig + cpu_MAC + cpu_smartc  # 主节点验证trans处理时间
    T_e = ver_mac_smc_time / state[primary_node_index + 9]  # 公式（16）处理一条交易的时间
    # print(max_req_t,T_e)
    T_q = (Sb / average_size_trans - 1) * T_e / 2  # 公式（17）平均队列延迟

    T_u = (max_req_t + T_e + T_q).__round__(6)  # 公式（18） 平均时延
    # print("平均时延：{}".format(T_u))

    # 区块链性能
    next_p_node_index = next_p_node(primary_node_index)  # 得到t+1period的主节点index
    throughput_all_block = (Sb * K) / (average_size_trans * T_dot)  # 公式（19）区块链系统吞吐量

    if primary_node_index == next_p_node_index:
        IB = 0  # 公式（21）
    else:
        ib_rate = trans_rate(action[primary_node_index + 3], Band_width, state_m[primary_node_index - 1][next_p_node_index - 1])
        # IB = ((Sb / ib_rate)/ (T_dot / K)) - 1   作者的算法   公式（21）
        IB = K*(1-((Sb / ib_rate)/ (T_dot / K)))
        IB = int(IB) if IB>0 else 0
    # print("IB：{}    K:{}  ".format(IB, K))  # 区块传输中忽略区块的个数

    # 共识算法吞吐量
    consensus_throutput = (throughput_all_block / K) * (K - IB)
    # print("区块链吞吐量：{}   共识算法吞吐量：{}".format(throughput_all_block, consensus_throutput))

    #  -------------------获取next_state-------------------
    current_p_node_index = next_p_node_index  # 新状态下主节点index
    next_state = np.zeros(14)
    next_state[0] = current_p_node_index
    for i in range(1, 10):  # 根据矩阵（38）更新各个节点的SNR状态
        # 获取当前状态在状态空间中的索引
        current_state_index = snr_values.index(state[i])
        # 通过状态转移概率矩阵计算下一个状态
        next_state_probs = state_trans_p_m[current_state_index]
        # 根据概率选择下一个状态
        next_state[i] = np.random.choice(snr_values, p=next_state_probs)
    for j in range(10, 14):
        current_compute_index = compute_resources.index(state[j])
        next_comstate_probs = compute_trans_p_m[current_compute_index]
        next_state[j] = np.random.choice(compute_resources, p=next_comstate_probs)

    # print("下一阶段state:{}".format(next_state)
# --------------------------------------------------------
    if T_f <= 8 :
        reward = weight_blo_sys*(1/T_u)+weight_mec_sys*(consensus_throutput)
    else:
        reward = 0
    return next_state, reward


# for tex in range(10):
#     actions = action_space()
#     num = actions.shape[0]
#     # print(num)
#     action = random.choice(actions)
#     n,r = step(init_state(), action)
#     for tex_1 in range(1000):
#         action = random.choice(actions)
#         n1,r1 = step(n, action)
#         n = n1
