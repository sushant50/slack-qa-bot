import xml.etree.ElementTree as ET

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

with open('data/pythondev/2019/merged-pythondev-help.xml', 'r') as f:
    xml_data = f.read()

conversations = parse_slack_data(xml_data)
import json

with open('conversations.json', 'w') as f:
    json.dump(conversations, f)

processed_conversations = prepare_data_for_model(conversations)


with open('processed_conversations.json', 'w') as f:
    json.dump(processed_conversations, f)


from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Now you can use tokenizer with pad_to_max_length parameter
# tokenized_conversations = {k: tokenizer.encode(v, padding='max_length', max_length=512) for k, v in processed_conversations.items()}
tokenized_conversations = {
    k: tokenizer.encode(v, padding='max_length', max_length=512)
    for k, v in processed_conversations.items()
}
from torch.utils.data import Dataset

# class ConversationDataset(Dataset):
#     def __init__(self, conversations):
#         self.conversations = conversations

#     def __len__(self):
#         return len(self.conversations)
    

#     def __getitem__(self, idx): return {
#         "input_ids": self.conversations[idx], 
#         "attention_mask": [0 if token == tokenizer.pad_token_id else 1 for token in self.conversations[idx]]
#     }

#     # def __getitem__(self, idx):
#     #     return {"input_ids": self.conversations[idx], "attention_mask": [1]*len(self.conversations[idx])}

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

# dataset = ConversationDataset(list(tokenized_conversations.values()))
dataset = ConversationDataset(
    list(tokenized_conversations.values()),
    tokenizer=tokenizer,
    max_length=512
)

from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import torch.nn as nn

class GPT2LMHeadModelWithLoss(GPT2LMHeadModel):
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        
        if self.training:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Flatten the logits and labels
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_labels = shift_labels.view(-1)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(flat_shift_logits, flat_shift_labels)

            return {"loss": loss}  # Return loss as part of a dictionary
        else:
            return outputs

model = GPT2LMHeadModelWithLoss.from_pretrained('gpt2')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
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


# Save the model
trainer.save_model('./my_model')

# Load the model and the tokenizer
model = GPT2LMHeadModelWithLoss.from_pretrained('./my_model')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
import torch

# Define a function to generate a response
def get_model_response(prompt):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate a response
    with torch.no_grad():
        output = model.generate(input_ids, max_length=1000, num_return_sequences=1, do_sample=True, temperature=0.7)
        
    # Decode the response
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response

# Ask a question
prompt = "PythonDev: How do I install Django?"
print(get_model_response(prompt))
