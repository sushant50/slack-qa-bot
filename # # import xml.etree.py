# # import xml.etree.ElementTree as ET
# # import json
# # import torch
# # from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# # from torch.utils.data import Dataset
# # import torch.nn as nn

# # # Check if a GPU is available and set the device accordingly
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # def parse_slack_data(xml_data):
# #     root = ET.fromstring(xml_data)
# #     conversations = {}
    
# #     for message in root.iter('message'):
# #         conversation_id = message.get('conversation_id')
# #         ts = message.find('ts').text
# #         user = message.find('user').text
# #         text = message.find('text').text

# #         if conversation_id not in conversations:
# #             conversations[conversation_id] = []

# #         conversations[conversation_id].append({
# #             'timestamp': ts,
# #             'user': user,
# #             'text': text
# #         })

# #     return conversations

# # def prepare_data_for_model(conversations):
# #     processed_conversations = {}

# #     for conv_id, messages in conversations.items():
# #         sorted_messages = sorted(messages, key=lambda x: x['timestamp'])
# #         texts = [f"{msg['user']}: {msg['text']}" for msg in sorted_messages]
# #         processed_conversations[conv_id] = " [SEP] ".join(texts)

# #     return processed_conversations

# # with open('merged-pythondev-help.xml', 'r') as f:
# #     xml_data = f.read()

# # conversations = parse_slack_data(xml_data)

# # with open('conversations.json', 'w') as f:
# #     json.dump(conversations, f)

# # processed_conversations = prepare_data_for_model(conversations)

# # with open('processed_conversations.json', 'w') as f:
# #     json.dump(processed_conversations, f)

# # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# # tokenizer.pad_token = tokenizer.eos_token

# # tokenized_conversations = {
# #     k: tokenizer.encode(v, padding='max_length', max_length=512)
# #     for k, v in processed_conversations.items()
# # }

# # class ConversationDataset(Dataset):
# #     def __init__(self, conversations, tokenizer, max_length):
# #         self.conversations = conversations
# #         self.tokenizer = tokenizer
# #         self.max_length = max_length

# #     def __len__(self):
# #         return len(self.conversations)

# #     def __getitem__(self, idx):
# #         conversation = self.conversations[idx]
# #         encoded_inputs = self.tokenizer.encode_plus(
# #             conversation,
# #             padding='max_length',
# #             max_length=self.max_length,
# #             truncation=True,
# #             return_tensors='pt'
# #         )

# #         input_ids = encoded_inputs['input_ids'].squeeze().to(device)
# #         attention_mask = encoded_inputs['attention_mask'].squeeze().to(device)

# #         return {
# #             "input_ids": input_ids,
# #             "attention_mask": attention_mask
# #         }

# # dataset = ConversationDataset(
# #     list(tokenized_conversations.values()),
# #     tokenizer=tokenizer,
# #     max_length=512
# # )

# # class GPT2LMHeadModelWithLoss(GPT2LMHeadModel):
# #     def forward(self, input_ids, attention_mask=None, **kwargs):
# #         input_ids = input_ids.to(device)
# #         attention_mask = attention_mask.to(device)

# #         outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)

# #         if self.training:
# #             shift_logits = outputs.logits[..., :-1, :].contiguous()
# #             shift_labels = input_ids[..., 1:].contiguous()

# #             flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
# #             flat_shift_labels = shift_labels.view(-1)

# #             loss_fct = nn.CrossEntropyLoss()
# #             loss = loss_fct(flat_shift_logits, flat_shift_labels)

# #             return {"loss": loss}
# #         else:
# #             return outputs

# # model = GPT2LMHeadModelWithLoss.from_pretrained('gpt2').to(device)

# # training_args = TrainingArguments(
# #     output_dir='./results',
# #     num_train_epochs=1,
# #     per_device_train_batch_size=1,
# #     per_device_eval_batch_size=1,
# #     warmup_steps=500,
# #     weight_decay=0.01,
# #     logging_dir='./logs',
# #     device=device
# # )

# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=dataset,
# # )

# # trainer.train()

# # trainer.save_model('./my_model')

# # model = GPT2LMHeadModelWithLoss.from_pretrained('./my_model').to(device)
# # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # def get_model_response(prompt):
# #     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# #     with torch.no_grad():
# #         output = model.generate(
# #             input_ids,
# #             max_length=1000,
# #             num_return_sequences=1,
# #             do_sample=True,
# #             temperature=0.7
# #         )

# #     response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

# #     return response

# # prompt = "PythonDev: How do I install Django?"
# # print(get_model_response(prompt))


# import xml.etree.ElementTree as ET
# import json
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# from torch.utils.data import Dataset
# import torch.nn as nn

# # Check if a GPU is available and set the device accordingly
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def parse_slack_data(xml_data):
#     root = ET.fromstring(xml_data)
#     conversations = {}
    
#     for message in root.iter('message'):
#         conversation_id = message.get('conversation_id')
#         ts = message.find('ts').text
#         user = message.find('user').text
#         text = message.find('text').text

#         if conversation_id not in conversations:
#             conversations[conversation_id] = []

#         conversations[conversation_id].append({
#             'timestamp': ts,
#             'user': user,
#             'text': text
#         })

#     return conversations

# def prepare_data_for_model(conversations):
#     processed_conversations = {}

#     for conv_id, messages in conversations.items():
#         sorted_messages = sorted(messages, key=lambda x: x['timestamp'])
#         texts = [f"{msg['user']}: {msg['text']}" for msg in sorted_messages]
#         processed_conversations[conv_id] = " [SEP] ".join(texts)

#     return processed_conversations

# with open('merged-pythondev-help.xml', 'r') as f:
#     xml_data = f.read()

# conversations = parse_slack_data(xml_data)

# with open('conversations.json', 'w') as f:
#     json.dump(conversations, f)

# processed_conversations = prepare_data_for_model(conversations)

# with open('processed_conversations.json', 'w') as f:
#     json.dump(processed_conversations, f)

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token

# tokenized_conversations = {
#     k: tokenizer.encode(v, padding='max_length', max_length=512)
#     for k, v in processed_conversations.items()
# }

# class ConversationDataset(Dataset):
#     def __init__(self, conversations, tokenizer, max_length):
#         self.conversations = conversations
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.conversations)

#     def __getitem__(self, idx):
#         conversation = self.conversations[idx]
#         encoded_inputs = self.tokenizer.encode_plus(
#             conversation,
#             padding='max_length',
#             max_length=self.max_length,
#             truncation=True,
#             return_tensors='pt'
#         )

#         input_ids = encoded_inputs['input_ids'].squeeze().to(device)
#         attention_mask = encoded_inputs['attention_mask'].squeeze().to(device)

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask
#         }

# dataset = ConversationDataset(
#     list(tokenized_conversations.values()),
#     tokenizer=tokenizer,
#     max_length=512
# )

# class GPT2LMHeadModelWithLoss(GPT2LMHeadModel):
#     def forward(self, input_ids, attention_mask=None, **kwargs):
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)

#         outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)

#         if self.training:
#             shift_logits = outputs.logits[..., :-1, :].contiguous()
#             shift_labels = input_ids[..., 1:].contiguous()

#             flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
#             flat_shift_labels = shift_labels.view(-1)

#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(flat_shift_logits, flat_shift_labels)

#             return {"loss": loss}
#         else:
#             return outputs

# model = GPT2LMHeadModelWithLoss.from_pretrained('gpt2').to(device)

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
# )

# # Set the device for the trainer
# trainer.model.device = device

# trainer.train()

# trainer.save_model('./my_model')

# model = GPT2LMHeadModelWithLoss.from_pretrained('./my_model').to(device)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# def get_model_response(prompt):
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

#     with torch.no_grad():
#         output = model.generate(
#             input_ids,
#             max_length=1000,
#             num_return_sequences=1,
#             do_sample=True,
#             temperature=0.7
#         )

#     response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

#     return response

# prompt = "PythonDev: How do I install Django?"
# print(get_model_response(prompt))

import xml.etree.ElementTree as ET
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch.nn as nn
pin_memory = False
# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_slack_data(xml_data):
    root = ET.fromstring(xml_data)
    conversations = {}
    
    for message in root.iter('message'):
        conversation_id = message.get('conversation_id')
        ts = message.find('ts').text
        user = message.find('user').text
        text = message.find('text').text

        if conversation_id not in conversations:
            conversations[conversation_id] = []

        conversations[conversation_id].append({
            'timestamp': ts,
            'user': user,
            'text': text
        })

    return conversations

def prepare_data_for_model(conversations):
    processed_conversations = {}

    for conv_id, messages in conversations.items():
        sorted_messages = sorted(messages, key=lambda x: x['timestamp'])
        texts = [f"{msg['user']}: {msg['text']}" for msg in sorted_messages]
        processed_conversations[conv_id] = " [SEP] ".join(texts)

    return processed_conversations

with open('merged-pythondev-help.xml', 'r',  encoding='utf-8') as f:
    xml_data = f.read()

conversations = parse_slack_data(xml_data)

with open('conversations.json', 'w') as f:
    json.dump(conversations, f)

processed_conversations = prepare_data_for_model(conversations)

with open('processed_conversations.json', 'w') as f:
    json.dump(processed_conversations, f)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

tokenized_conversations = {
    k: tokenizer.encode(v, padding='max_length', max_length=512)
    for k, v in processed_conversations.items()
}

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        encoded_inputs = self.tokenizer.encode_plus(
            conversation,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

dataset = ConversationDataset(
    list(tokenized_conversations.values()),
    tokenizer=tokenizer,
    max_length=512
)

class GPT2LMHeadModelWithLoss(GPT2LMHeadModel):
    def forward(self, input_ids, attention_mask=None, **kwargs):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)

        if self.training:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_labels = shift_labels.view(-1)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(flat_shift_logits, flat_shift_labels)

            return {"loss": loss}
        else:
            return outputs

model = GPT2LMHeadModelWithLoss.from_pretrained('gpt2')

# Move the model to the desired device
model = model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model('./my_model')

# Load the model and the tokenizer
model = GPT2LMHeadModelWithLoss.from_pretrained('./my_model')

# Move the model to the desired device
model = model.to(device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def get_model_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=1000,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response

prompt = "PythonDev: How do I install Django?"
print(get_model_response(prompt))


