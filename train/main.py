from dataset import DatasetFactory
from PreprocessingLoader import PreprocessingLoader
from ModelLoader import ModelFactory
from trainFactory import TrainFactory
from transformers import TrainingArguments
import evaluate
import torch
from HyperParameterFactory import HyperParameterFactory

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def main():
   
    try:
        dataset_loader = DatasetFactory.get_dataset("vision_transformer")
        dataset = dataset_loader.get_dataset()
    except ValueError as e:
        print(f"Error: {e}")
        return 
    
    splits = dataset.train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]

    
    model_checkpoint = "google/vit-base-patch16-224-in21k"
    preprocessing_loader = PreprocessingLoader(model_checkpoint)
    preprocessor = preprocessing_loader.get_preprocessor()
    train_ds.set_transform(preprocessor.preprocess_train)
    val_ds.set_transform(preprocessor.preprocess_val)

   
    model_loader = ModelFactory.create("vision_transformer")
    model = model_loader.get_model()

    print("Before PEFT:")
    print_trainable_parameters(model)

    from peft import get_peft_model

    peft_params = HyperParameterFactory.get_peft_parameters("lora")
    lora_model = get_peft_model(model, **peft_params)

  
    print("After PEFT:")
    print_trainable_parameters(lora_model)

    model_name = model_checkpoint.split("/")[-1]
    batch_size = 128
    args = TrainingArguments(
        f"{model_name}-finetuned-lora-food101",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        num_train_epochs=10,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        label_names=["labels"],
    )
 
    trainer = TrainFactory.get_trainer(lora_model, args, train_ds, val_ds, None, compute_metrics, None)

    train_results = trainer.train()


if __name__ == "__main__":
    main()
