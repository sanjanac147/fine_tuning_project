from transformers import Trainer
import torch
metric = evaluate.load("accuracy")

class TrainLoader:
    @staticmethod
    def compute_metrics(eval_pred, metric):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    @staticmethod
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, metric):
        self.trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=lambda eval_pred: self.compute_metrics(eval_pred, metric),
            data_collator=self.collate_fn,
        )

    def train(self):
        return self.trainer.train()

    def evaluate(self, val_ds):
        return self.trainer.evaluate(val_ds)

    def get_trainer(self):
        return self.trainer
