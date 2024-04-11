from transformers import TrainingArguments
from TrainFactory import TrainFactory
from dataset import DatasetFactory
from preprocessing.PreprocessingFactory import ImagePreprocessor
from model.ModelLoader import ModelFactory
import evaluate

def main():

    try:
        dataset_loader = DatasetFactory.get_dataset("vision_transformer")
        dataset = dataset_loader.get_dataset()
        
        splits = dataset.train_test_split(test_size=0.1)
        train_ds = splits["train"]
        val_ds = splits["test"]

        model_checkpoint = "google/vit-base-patch16-224-in21k"
        preprocessing_loader = ImagePreprocessor(model_checkpoint)
        preprocessor = preprocessing_loader.get_preprocessor()
        train_ds.set_transform(preprocessor.preprocess_train)
        val_ds.set_transform(preprocessor.preprocess_val)
    except ValueError as e:
        print(f"Error: {e}")
        return

    model_loader = ModelFactory.create("vision_transformer")
    model = model_loader.get_model()

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
    
    trainer = TrainFactory.get_trainer(model, args, train_ds, val_ds, preprocessor.get_tokenizer())

    train_results = trainer.train()
    print(train_results)

    evaluation_results = trainer.evaluate(val_ds)
    print(evaluation_results)

if __name__ == "__main__":
    main()
