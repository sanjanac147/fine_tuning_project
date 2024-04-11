import sys
sys.path.append('S:/SiDhU/Codes/HP/actual_repo/fine_tuning_project/')
import dataset
import model
import hyper_parameters
import train
import preprocessing
from preprocessing import initialize_image_processor
import peft_techniques
model_name = "vision_transformer"
model_checkpoint = "google/vit-base-patch16-224-in21k"
data = dataset.DatasetFactory.get_dataset(model_name)
parameter = {
            "r": 16,
            "lora_alpha": 16,
            "target_modules": ["query", "value"], 
            "lora_dropout": 0.1,
            "bias": "none",
        }
gen_params = {
            "batch_size": 128,
            "learning_rate": 0.01,
            "num_train_epochs": 2,
        }

peft_params = hyper_parameters.HyperParameterFactory.get_peft_parameters("lora",parameter)
general_params = hyper_parameters.HyperParameterFactory.get_general_parameters(gen_params)
print(peft_params)
print(general_params)
vision_model = model.ModelFactory.create(model_name, data)
processor = preprocessing.PreprocessingFactory(model_checkpoint)
train_ds, val_ds = processor.get_image_preprocessor(data)

# --------------------------------------------------------------------------------
# We have done till here 
# --------------------------------------------------------------------------------
# This we have to add to the peft_techniques
# --------------------------------------------------------------------------------

#check this.. since there is sep file for loraconfig 
from peft import LoraConfig, get_peft_model
lora_config= peft_techniques.LoRA.loadConfig(peft_params)
for key, value in peft_params.items():
    if hasattr(lora_config, key): 
        setattr(lora_config, key, value)
lora_model=get_peft_model(vision_model, lora_config)


#---------------------------------------------------------------------------------
from peft import LoraConfig, get_peft_model
for key, value in parameter.items():
    if hasattr(config, key): 
        setattr(config, key, value)
lora_model = get_peft_model(model, config)

# ----------------------------------------------------------------------------
# This fucntion I think we can add to the evaluation modules
# ----------------------------------------------------------------------------

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

print_trainable_parameters(lora_model)
# --------------------------------------------------------------------------------
# This I thought we have defined in the train module but in that it is directly pickking up 
# we need to figure out where to add this
# --------------------------------------------------------------------------------

# from this training args I have defined "batch_size": 128 "learning_rate": 0.01, "num_train_epochs": 2, which we will
# get from the user rest other parameter letter we can get from the user

# --------------------------------------------------------------------------------

from transformers import TrainingArguments, Trainer


model_name = model_checkpoint.split("/")[-1]

args =TrainingArguments(
    f"{model_name}-finetuned-lora-food101",
    **general_params
)

import numpy as np
import evaluate
image_processor = initialize_image_processor(model_checkpoint)
trainer = TrainFactory.get_trainer(lora_model, args, train_ds, val_ds, image_processor)

train_results = trainer.train()
print(train_results)

validation_results = trainer.evaluate(val_ds)


# That's it peace 🥳🥳
