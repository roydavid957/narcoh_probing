# -*- coding: utf-8 -*-
import torch
import argparse
import numpy as np
from zipfile import ZipFile
from transformers import AutoTokenizer, AutoModel, AutoConfig
from utils import load_all_samples, train_eval

'''
Adaptation of code by: 
https://github.com/wietsedv/bertje/blob/master/probing/probe.py
and
https://github.com/irenedini/tlink_probing/blob/main/code/roberta_probing.py
'''

def compute_embedding_mean(all_embeddings, start_idx, end_idx):
  '''
  Given all the sentence embeddings, computes the mean of sub-tokens embeddings from stard_idx to end_idx.
  Is used to compute event and sentence representations. 
  '''
  if start_idx == end_idx:
    return all_embeddings[start_idx]
  else:
    selected_embeddings = all_embeddings[start_idx:end_idx+1]
    return torch.mean(selected_embeddings, 0)

def extract_samples_representations(samples,args):

  tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
  model = AutoModel.from_pretrained(args.model_ckpt).to(args.device)
  num_hidden_layers = model.config.num_hidden_layers

  for sample in samples:

    if args.use_sentence_embeddings:
      encoded_input = [tokenizer(sample.input_sent, padding=True, truncation=True, return_tensors="pt").to(args.device)]
      input_token_ids = [np.where(np.array(encoded_input[0].word_ids()) == event_idx)[0] for event_idx in (sample.input_event_idx)]
    else:
      encoded_input = [tokenizer(sent.text, padding=True, truncation=True, return_tensors="pt").to(args.device) for sent in sample.input]
      input_token_ids = [np.where(np.array(enc_inp.word_ids()) == sent.event)[0] if 0<= sent.event <= len(sent.tokens) else np.where(np.array(enc_inp.word_ids()) == sent.event-1)[0] for enc_inp,sent in zip(encoded_input,sample.input)]

    encoded_target = tokenizer(sample.target.text, padding=True, truncation=True, return_tensors="pt").to(args.device)

    target_token_ids = np.where(np.array(encoded_target.word_ids()) == sample.target.event)[0]    # get the sub-tokens corresponding to the original
                                                                                                      # token of each event.

    with torch.no_grad():                                                                              
      model_output_1 = [model(**enc_inp, output_hidden_states=True) for enc_inp in encoded_input]
      model_output_2 = model(**encoded_target, output_hidden_states=True)
      hidden_states_1 = [model_output.hidden_states for model_output in model_output_1]
      hidden_states_2 = model_output_2.hidden_states  # is the input embedding layer
      for layer in range(1, num_hidden_layers+1):     # For each of the other layers we extract sample representations
          layer_output_1 = [torch.squeeze(hidden_states[layer]) for hidden_states in hidden_states_1]
          layer_output_2 = torch.squeeze(hidden_states_2[layer])

          # To compute event representations, we compute the mean of all event's sub-tokens embeddings
          input_event_embedding = [compute_embedding_mean(layer_output, input_token_id[0], input_token_id[-1]).cpu().detach().numpy() for layer_output,input_token_id in zip(layer_output_1,input_token_ids)]
          target_event_embedding = compute_embedding_mean(layer_output_2, target_token_ids[0], target_token_ids[-1]).cpu().detach().numpy()
          # To compute sentence representation, we compute the mean of all sentence subtokens
          input_embedding = [torch.mean(layer_output, 0).cpu().detach().numpy() for layer_output in layer_output_1]
          target_embedding = torch.mean(layer_output_2, 0).cpu().detach().numpy()

          sample.representation[layer] = {'input_embedding': input_embedding, 'input_event_embedding': input_event_embedding, 'target_embedding': target_embedding, 'target_event_embedding': target_event_embedding}
      

def create_features_vectors(samples,layer,args):
  features = []
  labels = []
  for sample in samples:
    embeddings = sample.representation[layer]
    if len(embeddings[f"input_{args.key}"])>1:  # concatenate representation of each input sample and the target sample
      if args.data_set.lower() == 'cmcnc': 
        concat_embed_list = np.concatenate(embeddings[f"input_{args.key}"]+[embeddings[f"target_{args.key}"]])
        padded_concat_embed_list = np.pad(concat_embed_list, (0, args.max_len - len(concat_embed_list)), 'constant')
        features.append(padded_concat_embed_list)                                                                     # if padding is needed, i.e. for CMCNC dataset
      else:
        features.append(np.concatenate(embeddings[f"input_{args.key}"]+[embeddings[f"target_{args.key}"]]))           # if no padding is needed, i.e. no varying input lenghts
    else:                                  # concatenate representation of the input and the target sample
      features.append(np.concatenate([embeddings[f"input_{args.key}"][0], embeddings[f"target_{args.key}"]]))
    labels.append(sample.label)
  return features, labels


def main():
    parser = argparse.ArgumentParser(description='Probing PTLMs for narrative coherence')
    parser.add_argument('-m', '--model_ckpt',
                      help='the name of the model (checkpoint) used for extracting representations.')
    parser.add_argument('-tr', '--train_file')
    parser.add_argument('-ts', '--test_file', default='', help='Leave empty for k-fold cv')
    parser.add_argument('-ds', '--data_set', help='Specify dataset for dataloader: "SCT" (Story Cloze Task), "NCT" (Narrative Cloze Task), "CMCNC" (Coherent Multiple Choice Narrative Cloze)')
    parser.add_argument('-o', '--output_file',
                      help='absolute path to output file for classification report')
    parser.add_argument('-l', '--lang',
                      help='en/nl for spacy model for event extraction', default='en')
    parser.add_argument('-od', '--output_dir', 
                      help='absolute path to output directory (including last "/")', default='')
    parser.add_argument('-p', '--output_prob', default=False)
    parser.add_argument('-e', '--use_event_embeddings', action='store_true', default=False)
    parser.add_argument('-s', '--use_sentence_embeddings', action='store_true', default=True)
    
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.key = 'event_embedding' if args.use_event_embeddings else 'embedding'

    print('\n\nUsing sentence Embeddings:',args.use_sentence_embeddings)
    print('\nUsing event Embeddings:', args.use_event_embeddings)
    print('\nUsing device:', args.device)
    print('\n\n')

    spacy_model = "en_core_web_sm" if args.lang == 'en' else "nl_core_news_sm"
    train_samples, labels_list = load_all_samples(args.train_file, args, spacy_model)        # Loading of train and test samples from
    if args.test_file == '':
       valid_samples, test_label_list = [], labels_list # empty for k-fold cv
    else:
      valid_samples, test_label_list = load_all_samples(args.test_file, args, spacy_model)   # train and test source files
    all_labels = list(set(test_label_list).union(set(labels_list)))

    model_config = AutoConfig.from_pretrained(args.model_ckpt)    
    args.num_hidden_layers = model_config.num_hidden_layers

    extract_samples_representations(train_samples, args)    # Computation of samples representation using contextual embeddings
    if valid_samples:
       extract_samples_representations(valid_samples, args)

    max_len_dev = max([len(np.concatenate(x.representation[1][f"input_{args.key}"]+[x.representation[1][f"target_{args.key}"]])) for x in train_samples])                           # get the max length
    max_len_test = max([len(np.concatenate(x.representation[1][f"input_{args.key}"]+[x.representation[1][f"target_{args.key}"]])) for x in valid_samples]) if valid_samples else 0  # for padding the (CMCNC) inputs 
    args.max_len = max_len_dev if max_len_dev >= max_len_test else max_len_test                                                                                                     # for the SVM inputs

    all_labels += ['macro avg', 'weighted avg', 'accuracy']
    
    model_config = AutoConfig.from_pretrained(args.model_ckpt)    
    num_hidden_layers = model_config.num_hidden_layers 
    layers_scores = {label:[] for label in all_labels}

    for layer in range(1, num_hidden_layers+1):                 # We execute the probing extracting representation from each layer
        print(f'\n\n------ layer {layer} ------\n')
        X_train, y_train = create_features_vectors(train_samples, layer, args)      # Creation of SVM inputs
        if valid_samples:
          X_test, y_test = create_features_vectors(valid_samples, layer, args)
        else:
          X_test, y_test = [],[]  # empty for k-fold cv
        cls_report, y_pred_proba = train_eval(X_train, y_train, X_test, y_test, prob=args.output_prob)        # SVM training and evaluation
        print(cls_report)

        if y_pred_proba:
          # Change the predicted output format
          res_dict = dict()
          for sample, y in zip(valid_samples,y_pred_proba):
              if sample.id in res_dict.keys():                                # Per ID, we get the
                  res_dict[sample.id][str(sample.label)].append(float(y[1]))  # probabilities of the correct vs. incorrect
              else:                                                           # final sequence, based on the probability
                  res_dict[sample.id] = {str(sample.label):[float(y[1])]}     # that an instance is predicted to be true
          # write reformatted output to zipped file      
          with open(f"{args.out_dir}answer.txt",'w') as f:
              f.write('InputStoryid,AnswerRightEnding\n')
              for idx,pred in res_dict.items(): 
                y = 1 if (sum(pred['1'])/len(pred['1'])) > (sum(pred['0'])/len(pred['0'])) else 2
                f.write(f"{idx},{y}\n")
          with ZipFile(f"{args.out_dir}layer_{layer}.zip", 'w') as myzip:
              myzip.write(f"{args.out_dir}answer.txt",'answer.txt')
        
        for label in all_labels:
          if label in cls_report:
            if label == 'accuracy':
              layers_scores[label].append(str(cls_report[label]))
            else:
              layers_scores[label].append(str(cls_report[label]['f1-score']))
          elif label == 'accuracy' and valid_samples == []:       # for cross validation scores
             layers_scores[label].append(str(cls_report.mean()))  # computes mean of k-fold cv per layer
          else:
              layers_scores[label].append(str(None))
    
    with open(args.output_file, 'w+') as out_file:                  # The output file will contain a row for each class f1-score and a row for 
        for label in layers_scores:                                 # macro avg, weighted avg and accuracy
            out_file.write(f'{[label]}\t{layers_scores[label]}\n')  # each row will contain a value for each layer

if __name__ == '__main__':
    main()


