import itertools
import math
import time
import numpy.core.numeric as _nx
import nevergrad as ng
import math
import torch  # this raises all the right alarm bells
from qlib.base.loss_transfer import TransferLoss
import qlib
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha360
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
import os
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord,PortAnaRecord



SPLIT_N = 20
def get_split_time(ary, indices_or_sections, df, axis=0):
    
    num_day = len(ary)
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
    

    def objective_function(time_division, dis_type = 'coral',):
        start = 0
        selected = [0, SPLIT_N]

        while len(selected) -2 < indices_or_sections -1:
        
            selected.extend(time_division)
            selected = sorted(selected)
                           
            dis_temp = 0
            for i in range(1, len(selected)-1):
                for j in range(i, len(selected)-1):
                    index_part1_start = start + sum_column[math.floor(selected[i-1] / SPLIT_N * num_day)] 
                    index_part1_end = start + sum_column[math.floor(selected[i] / SPLIT_N * num_day) if math.floor(selected[i] / SPLIT_N*num_day) != len(sum_column) else math.floor(selected[i] / SPLIT_N*num_day)-1]
                    feat_part1 = feat[index_part1_start: index_part1_end]
                    index_part2_start = start + sum_column[math.floor(selected[j] / SPLIT_N * num_day)]
                    index_part2_end = start + sum_column[math.floor(selected[j+1] / SPLIT_N * num_day) if math.floor(selected[j+1] / SPLIT_N * num_day) != len(sum_column) else math.floor(selected[j+1] / SPLIT_N * num_day)-1]
                    feat_part2 = feat[index_part2_start:index_part2_end]
                    criterion_transder = TransferLoss(loss_type= dis_type, input_dim=feat_part1.shape[1])
                    dis_temp += criterion_transder.compute(feat_part1, feat_part2)
            return -dis_temp.item()      
    # 除了不支持一维数据的其不能运行的优化器'BayesOptimBO','PCABO', 'BO',"VoronoiDE","PymooCMAES","PymooNSGA2","HyperOpt","DiscreteDoerrOnePlusOne",
    time_division = ng.p.Array(shape=(indices_or_sections-1,), lower=1, upper=SPLIT_N-1).set_integer_casting() 
    if indices_or_sections == 2:
        a = -0.069
    elif indices_or_sections == 3:
        a = -0.301
    elif indices_or_sections == 4:
        a = -0.390
    else:
        a = -1.068
    budget = SPLIT_N**(indices_or_sections-1)

    def sort_and_check_unique(x):
        sorted_x = sorted(x)
        seen = set()
        for item in sorted_x:
            if item in seen:
                # 如果发现重复元素，返回false表示未满足要求
                return False
            seen.add(item)
        if sorted_x != list(x):
            return False
        return True
    optimizer = ng.optimizers.QrDE(parametrization=time_division, budget=budget)
    optimizer.parametrization.register_cheap_constraint(sort_and_check_unique)
    start_time = time.time()
    early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.current_bests["minimum"].mean < a)
    optimizer.register_callback("ask", early_stopping)
    recommendation = optimizer.minimize(objective_function)  # best value
    end_time = time.time()
    # 计算函数的运行时间（以秒为单位）
    run_time = (end_time - start_time)
    loss = -objective_function(*recommendation.args, **recommendation.kwargs)
    selected = [0, SPLIT_N]
    selected.extend(list(recommendation.value))
    selected.sort()
    start = 0
    div_points = []
    num_day = len(ary)
    for i in range(indices_or_sections+1):
        index = start +  math.floor(selected[i] / SPLIT_N * num_day) 
        div_points.append(index)
    sub_arys = []  
    sary = _nx.swapaxes(ary, axis, 0)
    for i in range(indices_or_sections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))
    # 打开文件以写入模式（'w'表示写入）
    with open(r"C:\Users\hehailing\.conda\envs\nevergrad\Lib\site-packages\qlib\base\sub_arys.txt", "a") as file:
        # 使用循环遍历列表中的元素，并逐行写入文件
        for item in sub_arys:
            file.write(str(item) + "\n")
        # 打开文件以写入模式（'w'表示写入）
    with open(r"C:\Users\hehailing\.conda\envs\nevergrad\Lib\site-packages\qlib\base\loss.txt", "a") as file:
        file.write(str(indices_or_sections) + "\n")
        # 直接将浮点数写入文件
        file.write(str(loss) + "\n")

    with open(r"C:\Users\hehailing\.conda\envs\nevergrad\Lib\site-packages\qlib\base\run_time.txt", "a") as file:
        file.write(str(indices_or_sections)+ "\n")
        # 使用循环遍历列表中的元素，并逐行写入文件
        
        file.write(str(run_time) + "\n")
    return sub_arys


    


 
    