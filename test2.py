# Load the model and the tokenizer
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import torch.nn as nn
from transformers import GPT2Tokenizer

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
        
model = GPT2LMHeadModelWithLoss.from_pretrained('./my-model2')
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
prompt = "PythonDev: When should Flask be used"
print(get_model_response(prompt))
