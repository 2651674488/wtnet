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
        self.seq_len = pred_len
        self.pred_len = pred_len

        # model component
        self.activ = "gelu"
        self.norm = None
        self.wavelet = "db4"
        self.level = 3
        self.filter_len = 8
        self.axis = 1
        #
        # # loss
        self.loss_fn = 'mse'
        # self.lambda_acf = 0.5
        # self.lambda_mse = 0.1
        # self.acf_cutoff = 2
        #
        # # optim
        self.grad_clip_val = 1
        self.patience = 1
        self.lr = 1e-3
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
        self.in_chn = 7
        self.out_chn = pred_len
        self.drop = 0.5
        self.hid_chn = 512
        self.weight_decay = 1


class ETTh2_LTFConfig(LTFConfig):
    def __init__(self, pred_len) -> None:
        super().__init__(pred_len=pred_len)
        self.name = "ETTh2"


class ETTm1_LTFConfig(LTFConfig):
    def __init__(self, pred_len) -> None:
        super().__init__(pred_len=pred_len)
        self.name = "ETTm1"


class ETTm2_LTFConfig(LTFConfig):
    def __init__(self, pred_len) -> None:
        super().__init__(pred_len=pred_len)
        self.name = "ETTm2"
