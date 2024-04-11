from .TrainLoader import TrainLoader

class TrainFactory:
    @staticmethod
    def get_trainer(model, args, train_dataset, eval_dataset, tokenizer):
        return TrainLoader(model, args, train_dataset, eval_dataset, tokenizer)
