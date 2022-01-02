# %matplotlib inline
import glob
import numpy as np
import pandas as pd
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from qlib.workflow import R

import os

# os.chdir('examples/benchmarks/TRA')

sns.set(style='white')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from tqdm.auto import tqdm
from joblib import Parallel, delayed

def func(x, N=80):
    ret = x.ret.copy()
    x = x.rank(pct=True)
    x['ret'] = ret
    diff = x.score.sub(x.label)
    r = x.nlargest(N, columns='score').ret.mean()
    r -= x.nsmallest(N, columns='score').ret.mean()
    return pd.Series({
        'MSE': diff.pow(2).mean(), 
        'MAE': diff.abs().mean(), 
        'IC': x.score.corr(x.label),
        'R': r
    })
    
ret = pd.read_pickle("examples/benchmarks/TRA/data/ret.pkl").clip(-0.1, 0.1)
def backtest(fname, **kwargs):
    pred = pd.read_pickle(fname).loc['2018-09-21':'2020-06-30']  # test period
    pred['ret'] = ret
    dates = pred.index.unique(level=0)
    res = Parallel(n_jobs=-1)(delayed(func)(pred.loc[d], **kwargs) for d in dates)
    res = {
       dates[i]: res[i]
       for i in range(len(dates))
    }
    res = pd.DataFrame(res).T
    r = res['R'].copy()
    r.index = pd.to_datetime(r.index)
    r = r.reindex(pd.date_range(r.index[0], r.index[-1])).fillna(0)  # paper use 365 days
    return {
        'MSE': res['MSE'].mean(),
        'MAE': res['MAE'].mean(),
        'IC': res['IC'].mean(),
        'ICIR': res['IC'].mean()/res['IC'].std(),
        'AR': r.mean()*365,
        'AV': r.std()*365**0.5,
        'SR': r.mean()/r.std()*365**0.5,
        'MDD': (r.cumsum().cummax() - r.cumsum()).max()
    }, r

def fmt(x, p=3, scale=1, std=False):
    _fmt = '{:.%df}'%p
    string = _fmt.format((x.mean() if not isinstance(x, (float, np.floating)) else x) * scale)
    if std and len(x) > 1:
        string += ' ('+_fmt.format(x.std()*scale)+')'
    return string

def backtest_multi(files, **kwargs):
    res = []
    pnl = []
    for fname in files:
        metric, r = backtest(fname, **kwargs)
        res.append(metric)
        pnl.append(r)
    res = pd.DataFrame(res)
    pnl = pd.concat(pnl, axis=1)
    return {
        'MSE': fmt(res['MSE'], std=True),
        'MAE': fmt(res['MAE'], std=True),
        'IC': fmt(res['IC']),
        'ICIR': fmt(res['ICIR']),
        'AR': fmt(res['AR'], scale=100, p=1)+'%',
        'VR': fmt(res['AV'], scale=100, p=1)+'%',
        'SR': fmt(res['SR']),
        'MDD': fmt(res['MDD'], scale=100, p=1)+'%'
    }, pnl

exps = {
    'Linear': ['output/Linear/pred.pkl'],
    'LightGBM': ['output/GBDT/lr0.05_leaves128/pred.pkl'],
    'MLP': glob.glob('output/search/MLP/hs128_bs512_do0.3_lr0.001_seed*/pred.pkl'),
    'SFM': glob.glob('output/search/SFM/hs32_bs512_do0.5_lr0.001_seed*/pred.pkl'),
    'ALSTM': glob.glob('output/search/LSTM_Attn/hs256_bs1024_do0.1_lr0.0002_seed*/pred.pkl'),
    'Trans.': glob.glob('output/search/Transformer/head4_hs64_bs1024_do0.1_lr0.0002_seed*/pred.pkl'),
    'ALSTM+TS':glob.glob('output/LSTM_Attn_TS/hs256_bs1024_do0.1_lr0.0002_seed*/pred.pkl'),
    'Trans.+TS':glob.glob('output/Transformer_TS/head4_hs64_bs1024_do0.1_lr0.0002_seed*/pred.pkl'),
    'ALSTM+TRA(Ours)': glob.glob('output/search/finetune/LSTM_Attn_tra/K10_traHs16_traSrcLR_TPE_traLamb2.0_hs256_bs1024_do0.1_lr0.0001_seed*/pred.pkl'),
    'Trans.+TRA(Ours)': glob.glob('output/search/finetune/Transformer_tra/K3_traHs16_traSrcLR_TPE_traLamb1.0_head4_hs64_bs512_do0.1_lr0.0005_seed*/pred.pkl')
}



if __name__ == "__main__":
    # res = {
    # name: backtest_multi(exps[name])
    # for name in tqdm(exps)
    # }
    # report = pd.DataFrame({
    #     k: v[0]
    #     for k, v in res.items()
    # }).T
    # # report
    # print(report.to_latex())

    # recorder = R.get_recorder(recorder_id='617c5af3bbed450e8aaf0a182753868f', experiment_name="backtest_analysis")
    # print(recorder)
    # pred_df = recorder.load_object("pred.pkl")
    backtest_multi(['examples/mlruns/2/617c5af3bbed450e8aaf0a182753868f/artifacts/pred.pkl'])