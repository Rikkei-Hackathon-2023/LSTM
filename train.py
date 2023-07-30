from trainer.trainer import Trainer

if __name__ == "__main__":
    model_config = {
        "num_classes": 5,
        "input_size": 5,
        "hidden_size": 64,
        "num_layers": 2,
        "bi": True
    }
    data_config = {
        'csv_file': 'data_sol.csv',
        'time_span': 100,
        'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'scale': True,
        'only_last': True
    }
    train_config = {
        'batch_size': 16,
        'num_workers': 8,
        'device': 'cuda:0',
        'lr': 1e-4,
        'wd': 1e-2,
        'epoch': 50,
    }

    trainer = Trainer(model_config, data_config, train_config)
    trainer.train()
