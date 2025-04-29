class LTFConfig():

    def __init__(self, pred_len) -> None:
        self.task_name = "long-term-forecast"
        self.pred_len = pred_len
        self.embed = 'timeF'
        self.features = 'M'
        self.target = 'OT'
        self.label_len = 0
        self.batch_size = 16  # 32
        self.num_workers = 1  # 8
        self.freq = 'h'
        self.seq_len = 96
        self.pred_len = pred_len

        # model component
        self.activ = "gelu"
        self.norm = None
        self.wavelet = "db4"
        self.filter_len = 8
        self.level = 3
        self.axis = 1
        self.scale = True
        #
        # # loss
        self.loss_fn = 'mae'
        self.lambda_acf = 0.5
        self.lambda_mse = 0.1
        self.acf_cutoff = 2
        #
        # # optim
        self.grad_clip_val = 1
        self.patience = 1

        self.lr_factor = 0.5
        self.optim = "adamw"
        self.weight_decay = 0.1


class ETTh1_LTFConfig(LTFConfig):
    def __init__(self, pred_len) -> None:
        super().__init__(pred_len=pred_len)
        self.name = "etth1"
        self.data = "ETTh1"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTh1.csv"
        self.in_chn = 1
        self.out_chn = 1
        self.drop = 0.3
        self.hid_chn = 512
        self.in_seq = 7
        self.out_seq = pred_len
        self.hid_seq = 512
        self.weight_decay = 1
        self.lr = 1e-4  # 1e-4
        self.scale = True
        if self.features == 'M':
            self.out_chn = 7


class ETTh2_LTFConfig(LTFConfig):
    def __init__(self, pred_len) -> None:
        super().__init__(pred_len=pred_len)
        self.name = "etth2"
        self.data = "ETTh2"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTh2.csv"
        self.in_chn = 7
        self.out_chn = 1
        self.drop = 0.6
        self.hid_chn = 512
        self.in_seq = 7
        self.out_seq = pred_len
        self.hid_seq = 512
        self.weight_decay = 1
        self.lr = 1e-4  # 1e-4
        self.scale = True
        if self.features == 'M':
            self.out_chn = 7


class ETTm1_LTFConfig(LTFConfig):
    def __init__(self, pred_len) -> None:
        super().__init__(pred_len=pred_len)
        self.name = "ettm1"
        self.data = "ETTm1"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTm1.csv"
        self.in_chn = 7
        self.out_chn = 1
        self.drop = 0.5
        self.hid_chn = 512
        self.in_seq = 7
        self.out_seq = pred_len
        self.hid_seq = 512
        self.weight_decay = 1
        self.lr = 1e-3  # 1e-4
        self.scale = True

        if self.features == 'M':
            self.out_chn = 7


class ETTm2_LTFConfig(LTFConfig):
    def __init__(self, pred_len) -> None:
        super().__init__(pred_len=pred_len)
        self.name = "ettm2"
        self.data = "ETTm2"
        self.root_path = "./dataset/ETT-small/"
        self.data_path = "ETTm2.csv"
        self.drop = 0.5
        self.in_seq = 7
        self.out_seq = pred_len
        self.hid_seq = 512
        self.weight_decay = 1
        self.lr = 1e-3  # 1e-4
        self.scale = True
        if self.features == 'M':
            self.out_chn = 7


class ECL_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "ecl"
        self.data = "custom"
        self.root_path = "./dataset/electricity/"
        self.data_path = "electricity.csv"
        self.in_seq = pred_len * 2
        self.out_seq = pred_len
        self.hid_seq = 512
        self.drop = 0.5
        self.out_chn = 1
        self.lr = 1e-3  # 1e-4
        self.level = 3
        self.scale = True
        if self.features == 'M':
            self.out_chn = 321


class Traffic_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "traffic"
        self.data = "custom"
        self.root_path = "./dataset/traffic/"
        self.data_path = "traffic.csv"

        self.batch_size = 16
        self.in_seq = 7
        self.out_seq = pred_len
        self.hid_seq = 512
        self.drop = 0.5
        self.lr = 1e-3
        self.level = 3
        self.out_chn = 1
        self.scale = True
        if self.features == 'M':
            self.out_chn = 862


class Weather_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "weather"
        self.data = "custom"
        self.root_path = "./dataset/weather/"
        self.data_path = "weather.csv"
        self.in_seq = 7
        self.out_seq = pred_len
        self.hid_seq = 512
        self.drop = 0.0
        self.level = 3
        self.out_chn = 1
        self.lr = 1e-3
        self.scale = True
        if self.features == 'M':
            self.out_chn = 21


class Exchange_LTFConfig(LTFConfig):

    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "exchange"
        self.data = "custom"
        self.root_path = "./dataset/exchange_rate/"
        self.data_path = "exchange_rate.csv"
        # self.in_seq = 7
        self.out_seq = pred_len
        self.hid_seq = 512
        self.drop = 0.5
        self.level = 3
        self.out_chn = 1
        self.lr = 1e-4  # 1e-4
        self.scale = True
        if self.features == 'M':
            self.out_chn = 8
