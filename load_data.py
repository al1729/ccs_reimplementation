from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from tqdm import tqdm

def generate_amazon_prompt(review, label):
  '''
  review: text of review, from dataset
  label: 1 (+) or -1 (-)
  '''
  if label==1:
    label=' positive'
  else:
    label=' negative'
  prompt = "Consider the following example: '" + review + "' Between positive and negative, the sentiment of this example is:" + label
  return(prompt)

def get_hidden_states(model, tokenizer, input_text, layer=-1):
  '''
  model, tokenizer: pretrained GPT-J model and tokenizer
  input_text: input text from dataset
  layer: ID of layer
  '''
  input_text += tokenizer.eos_token # leads to gains when using
  tokenized_text = tokenizer(input_text, return_tensors="pt")
  text_ids = tokenized_text.input_ids.to(model.device)

  with torch.no_grad():
    model_output = model(text_ids, output_hidden_states = True)
    # -2 is used to get the embeddings of the +/- label
  hidden_states = model_output["hidden_states"][layer][0, -2] 
  return(hidden_states.detach().cpu().numpy())

def get_all_hidden_states(model, tokenizer, data, n=1000, embedding_size=4096):
  '''
  randomly sample n positive and n negative embeddings from data, then compute embeddings.
  '''
  model.eval()
  negative_hidden_states = np.zeros((n,embedding_size))
  positive_hidden_states = np.zeros((n,embedding_size))
  labels = np.zeros((n,))
  count = 0

  pbar = tqdm(total = n)
  while count < n:
    idx = np.random.randint(len(data))
    content, label = data[idx]["content"], data[idx]["label"]
    negative_hidden_states[count,:] = get_hidden_states(model, tokenizer, generate_amazon_prompt(content,-1))
    positive_hidden_states[count,:] = get_hidden_states(model, tokenizer, generate_amazon_prompt(content,1))
    labels[count] = label
    count += 1
    pbar.update(1)

  pbar.close()
  return(negative_hidden_states, positive_hidden_states, labels)

def save_hidden_states(hidden_states, data_size, train_split, fn=''):
    '''
    given a hidden_states array, shuffle pos and neg values and split into train and test sets to save.
    '''
    indices = list(range(data_size))
    np.random.shuffle(indices)

    # split into train and test
    train_len = int(train_split * data_size)
    train_indices, test_indices = indices[:train_len], indices[train_len:]
    neg_train, neg_test = hidden_states[0][train_indices], hidden_states[0][test_indices]
    pos_train, pos_test = hidden_states[1][train_indices], hidden_states[1][test_indices]

    # save data
    np.save(fn+'_neg_hidden_states_train.npy', neg_train)
    np.save(fn+'_neg_hidden_states_test.npy', neg_test)
    np.save(fn+'_pos_hidden_states_train.npy', pos_train)
    np.save(fn+'_pos_hidden_states_test.npy', pos_test)
    np.save(fn+'_labels.npy', hidden_states[2])

if __name__ == "__main__":
    data = load_dataset("amazon_polarity")["test"]
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

    if (torch.cuda.is_available()):
        model.cuda()
    data_size = 1000
    train_split = 0.6

    hidden_states = get_all_hidden_states(model, tokenizer, data, data_size)
    save_hidden_states(hidden_states, data_size, train_split)
    
