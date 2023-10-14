import math
import numpy.core.numeric as _nx

import qlib
import torch  # this raises all the right alarm bells
from qlib.base.loss_transfer import TransferLoss
from qlib.data import D
def get_split_time(ary, indices_or_sections, df, dis_type = 'coral', axis=0):
    split_N = 10
    num_day = len(ary)
    # feat = df['feature']
    # selected_columns = ["close","change","high","open","volume","factor","low"]
    # feat = feat[selected_columns]
    feat=torch.tensor(df.values, dtype=torch.float32)
    # 使用groupby和size函数来计算每个日期对应的instrument数量
    count_by_date = df.groupby('datetime').size().reset_index(name='count')
    # 创建一个空的列表，用于存储每个日期之前的第二列的和
    sum_column = []
    # 遍历日期列
    for index, row in count_by_date.iterrows():
        # 获取截止到当前日期的子数据框
        sub_df = count_by_date[count_by_date['datetime'] < row['datetime']]
        # 计算第二列的和，并添加到列表中
        sum_value = sub_df['count'].sum()
        sum_column.append(sum_value)
    sum_column.append(count_by_date.iloc[-1]["count"]+ sum_column[-1])
    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0

    if indices_or_sections in [2, 3, 4, 5, 7, 10]:
        while len(selected) -2 < indices_or_sections -1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected)-1):
                    for j in range(i, len(selected)-1):
                        index_part1_start = start + sum_column[math.floor(selected[i-1] / split_N * num_day)] 
                        index_part1_end = start + sum_column[math.floor(selected[i] / split_N * num_day) if math.floor(selected[i] / split_N*num_day) != len(sum_column) else math.floor(selected[i] / split_N*num_day)-1]
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + sum_column[math.floor(selected[j] / split_N * num_day)]
                        index_part2_end = start + sum_column[math.floor(selected[j+1] / split_N * num_day) if math.floor(selected[j+1] / split_N * num_day) != len(sum_column) else math.floor(selected[j+1] / split_N * num_day)-1]
                        feat_part2 = feat[index_part2_start:index_part2_end]
                        criterion_transder = TransferLoss(loss_type= dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(feat_part1, feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            # with open(r"C:\Users\jsu\.conda\envs\nevergrad\Lib\site-packages\qlib\base\loss.txt", "a") as file:
            #     file.write(str(max(distance_list)) + "\n")
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index]) 
        selected.sort()
        div_points = []
        for i in range(indices_or_sections+1):
            index = start +  math.floor(selected[i] / split_N * num_day) 
            div_points.append(index)
        sub_arys = []  
        sary = _nx.swapaxes(ary, axis, 0)
    for i in range(indices_or_sections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))

    return sub_arys