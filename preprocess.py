# -*- coding: utf-8 -*-

import pandas as pd
import spacy
import numpy as np
from tqdm.notebook import tqdm
from typing import Tuple
import argparse

def get_num_lines(path:str):
  return sum(1 for line in open(path))

def all_cmcnc_stories(path:str,n=5,s=2023) -> list[dict]:
  np.random.seed(s)

  with open(path,'r') as f:
    story_list_dict = []
    num_lines = get_num_lines(path)

    for line in tqdm(f,total=num_lines):
      if line.count('|SENT|') >= n-1:
        sentences = []
        story_id = line.split('|')[0]
        raw_story = [l for l in line.strip('\n').split('|SENT|')]

        for sentence in raw_story:
          sent_id = sentence.split('|')[1]
          sent = sentence.split('|')[2]
          tup = (',').join(sentence.split(sent)[-1].split('|TUP|')[1:])
          sentence_dict = {'SentID':sent_id,'sentence':sent,'tup':tup}
          sentences.append(sentence_dict)

        story_list_dict.append({'StoryID':story_id,
                                'Sentence1':sentences[0]['sentence'],'TUP1':sentences[0]['tup'],
                                'Sentence2':sentences[1]['sentence'],'TUP2':sentences[1]['tup'],
                                'Sentence3':sentences[2]['sentence'],'TUP3':sentences[2]['tup'],
                                'Sentence4':sentences[3]['sentence'],'TUP4':sentences[3]['tup'],
                                'Sentence5':sentences[4]['sentence'],'TUP5':sentences[4]['tup']})
  
  # add random final sentences
  picked_id_list = []             # include only 1 sentence per story
  for story in story_list_dict:   # to avoid possible final sentence bias
    add=True
    while add: 
      rand_sent = np.random.choice(story_list_dict)
      if rand_sent['StoryID'] not in picked_id_list and rand_sent['StoryID'] != story['StoryID']:
        n = np.random.choice([i+1 for i in range(n)])
        # story['RandomFinalStoryID'] = rand_sent['StoryID']
        story['RandomFinalSentence'] = rand_sent[f"Sentence{n}"]
        story['RandomFinalTUP'] = rand_sent[f"TUP{n}"]
        picked_id_list.append(rand_sent['StoryID'])
        add=False

  return story_list_dict

def get_prot(doc):
  nsubj = [str(tok) for tok in doc if tok.dep_ == "subj" in tok.dep_ or "obj" in tok.dep_ or "nmod" in tok.dep_]
  ents = [str(ent) for ent in doc.ents if ent.label_ == "PERSON"]
  if len(set(nsubj).intersection(set(ents))) > 0:
    return True
  else:
    return False

def get_prot_stories(data: pd.DataFrame,spacy_model="en_core_web_sm",s=2023) -> list[dict]:
  np.random.seed(s)
  nlp = spacy.load(spacy_model)
  
  story_list_dict = []
  for idx, row in tqdm(data.iterrows(),total=data.shape[0]):
    add=False
    story = [row['Sentence1'],row['Sentence2'],row['Sentence3'],row['Sentence4'],row['Sentence5']]
    for sentence in story:
      doc = nlp(sentence)
      if get_prot(doc):
        add=True
        break

    if add:
      story_list_dict.append(row)

  return story_list_dict

def load_cmcnc(src_path:str)->list[dict]:
  samples = []
  sample = []
  with open(src_path,'r') as f:
    num_lines = get_num_lines(src_path)
    for idx,line in tqdm(f,total=num_lines):
      line = line.strip('\n')
      if line == 'DOC_SEP':
        target_idx = sample.index('HOLDOUT_SEP')
        neg_sep_idx = sample.index('NEG_SEP')
        samples.append({'StoryID':str(idx),'input':('|').join(sample[:target_idx]),'target':sample[target_idx+1],'neg':('|').join(sample[neg_sep_idx+1:])})
        sample = []
      elif line:
        sample.append((' ').join(line.split('|')))
  return samples

def main():
    parser = argparse.ArgumentParser(description='Retrieves stories/event sequences, with correct and incorrect ending(s)')
    parser.add_argument('-f', '--file_name', help='absolute path to data file')
    parser.add_argument('-d', '--data_set', help='dataset name: story, event', default='story')
    parser.add_argument('-p', '--use_prot', help='only stories with protagonist, only applicable for story dataset', default=False)
    parser.add_argument('-o', '--output_file', help='absolute path to output file')

    args = parser.parse_args()

    print('\n\nUsing protagonist:',args.use_prot)
    print('\n\n')

    data = all_cmcnc_stories(args.file_name) if args.data_set == 'story' else load_cmcnc(args.file_name)
    df = pd.DataFrame(data)
    if args.use_prot and args.data_set == 'story':
      df = get_prot_stories(df)
    out_path = f"{args.output_file.split('.')[0]}.tsv"
    df.to_csv(out_path,index=False,sep='\t')

if __name__ == '__main__':
    main()

