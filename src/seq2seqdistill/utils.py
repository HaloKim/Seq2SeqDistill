from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import BartConfig, T5Config
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import torch


# def copy_and_freeze_embeddings(teacher_model, student_model):
#     if isinstance(teacher_model, BartForConditionalGeneration) and isinstance(student_model, BartForConditionalGeneration):
#         student_model.model.shared.weight = torch.nn.Parameter(teacher_model.model.shared.weight.clone())
#     elif isinstance(teacher_model, T5ForConditionalGeneration) and isinstance(student_model, T5ForConditionalGeneration):
#         student_model.shared.weight = torch.nn.Parameter(teacher_model.shared.weight.clone())
#     else:
#         raise ValueError('Model types must match and be either BART or T5')

#     # Freeze the embedding layer
#     for param in student_model.parameters():
#         param.requires_grad = True

#     if isinstance(student_model, BartForConditionalGeneration):
#         student_model.model.shared.weight.requires_grad = False
#     elif isinstance(student_model, T5ForConditionalGeneration):
#         student_model.shared.weight.requires_grad = False


def load_teacher_model(model_type: str, local_path: str, model_name: str) -> BartForConditionalGeneration or T5ForConditionalGeneration:
    if model_type == 'bart':
        #check if local path is provided
        if local_path:
            model = BartForConditionalGeneration.from_pretrained(local_path)
        else:
            model = BartForConditionalGeneration.from_pretrained(model_name)
    elif model_type == 't5':
        if local_path:
            model = T5ForConditionalGeneration.from_pretrained(local_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        raise ValueError('model_type must be bart or t5')
    return model

def load_student_model(model_type: str, num_encoder_layers: int, num_decoder_layers: int, hidden_dim: int, vocab_size: int):
    if model_type == 'bart':
        bart_config = BartConfig(
            d_model=hidden_dim,
            encoder_layers=num_encoder_layers,
            decoder_layers=num_decoder_layers,
            vocab_size=vocab_size,
        )
        student_model = BartForConditionalGeneration(bart_config)

    elif model_type == 't5':
        t5_config = T5Config(
            d_model=int(hidden_dim),
            encoder_layers=int(num_encoder_layers),
            decoder_layers=int(num_decoder_layers),
            vocab_size=int(vocab_size),
            decoder_start_token_id=0
        )
        student_model = T5ForConditionalGeneration(t5_config)
    else:
        raise ValueError('model_type must be bart or t5')

    return student_model

def load_tokenizer(model_type: str, local_path: str, model_name: str) -> AutoTokenizer:
    if model_type == 'bart':
        #check if local path is provided
        if local_path:
            tokenizer = AutoTokenizer.from_pretrained(local_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == 't5':
        if local_path:
            tokenizer = AutoTokenizer.from_pretrained(local_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError('both custom_tokenizer_local_path and teacher model cannot be None')
    return tokenizer

def load_distill_dataset(dataset_name: str, local_path: str, dataset_data_type: str):
        if local_path is not None:
            if dataset_data_type:
                dataset = load_dataset(dataset_data_type, data_files={'train': local_path+f"train.{dataset_data_type}",
"validation": local_path+f"valid.{dataset_data_type}"})
            else:
                raise ValueError('dataset_data_type must be provided while using local dataset path')
        elif dataset_name:
            dataset = load_dataset(dataset_name)
        else:
            raise ValueError('dataset_name or dataset_local_path must be provided') 
        
        # split dataset into train and validation
        # split_dataset = dataset['train'].train_test_split(test_size=0.1, shuffle=True)
        # train_dataset = split_dataset['train']
        # val_dataset = split_dataset['test']
        return dataset["train"], dataset["validation"]
