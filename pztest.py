import qlib
from qlib.contrib.data.handler import Alpha158

qlib.init(provider_uri='~/.qlib/qlib_data/cn_data') # 使用现成的那个库

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": "csi300",
}

h = Alpha158(**data_handler_config)

h.get_cols()