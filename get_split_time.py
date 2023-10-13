import itertools
import math
import numpy.core.numeric as _nx
import nevergrad as ng
import math
import torch  # this raises all the right alarm bells
from qlib.base.loss_transfer import TransferLoss

def generate_combinations(length):
    if length < 1 or length > 9:
        raise ValueError("长度必须介于 1 到 9 之间")

    numbers = list(range(1, 10))
    all_combinations = list(itertools.combinations(numbers, length))
    
    return all_combinations
def get_split_time(ary, indices_or_sections, df, axis=0):
    
    num_day = len(ary)
    split_N = 10
    feat = df['feature']
    selected_columns = ['CLOSE0', 'OPEN0','HIGH0','LOW0','VWAP0','VOLUME0']
    feat = feat[selected_columns]
    feat=torch.tensor(feat.values, dtype=torch.float32)
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

    def test(indices_or_sections, feat, sum_column, num_day, split_N , x, dis_type = 'coral',):
        # x = [1,2,4,5,6,7]
        # print(x)
        start = 0
        selected = [0, 10]

        if indices_or_sections in [2, 3, 5, 7, 10]:
            while len(selected) -2 < indices_or_sections -1:
                candidate = sorted(list(x))
                selected.extend(candidate)
                selected = sorted(selected)
                               
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
                return -dis_temp.item()      
    choices = generate_combinations(indices_or_sections-1)
    instrum = ng.p.Instrumentation(indices_or_sections, feat, sum_column, num_day, split_N,
    ng.p.TransitionChoice(choices=choices))
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100)
    recommendation = optimizer.minimize(test)
    
    selected = [0, 10]
    selected.extend(list(recommendation.value[0][5]))
    selected.sort()
    split_N = 10
    start = 0
    div_points = []
    num_day = len(ary)
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