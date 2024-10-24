from dataclasses import dataclass

@dataclass
class TrainConfig:
    root : str = 'Dataset/machine_translation_daily_dialog_en_fa/data/'
    save_as = 'model.pth'

    n_epochs : int = 1000

    batch_size : int = 64

    lr : float= 5e-5
    weight_decay : float = 0.01

    monitor_value : str = 'test_loss'
    monitor_mode : str = 'min'
    monitor_delta : int = 0.01