import re
import pandas as pd
import spacy
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

def get_verbs(doc,vb=False)->list:
  if vb:
      return [f"{tok}|{tok.lemma_}|{tok.dep_}|{tok.idx}|{tok.idx+len(tok)}|{tok.pos_}|{tok.tag_}" for tok in doc if tok.pos_ == "VERB" or "VB" in tok.tag_]
  return [f"{tok}|{tok.lemma_}|{tok.dep_}|{tok.idx}|{tok.idx+len(tok)}|{tok.pos_}|{tok.tag_}" for tok in doc if tok.pos_ == "VERB" or "VB" in tok.tag_ or "amod" in tok.dep_ or "prep" in tok.dep_ or "UH" in tok.tag_ or "ROOT" in tok.dep_]

def get_word_idx(sent):
  offset=0
  tokens = []
  for word in sent.split(' '):
    tokens.append(f"{word}|{offset}|{offset+len(word)}")
    offset+=len(word)+1
  return tokens

def get_event(verbs:list)->str:
  if len(verbs) == 1:
    return verbs[0]
  else:
    event = [verb for verb in verbs if verb.split("|")[2]=="ROOT" and verb.split("|")[5]=="VERB"]
    if event:
      return event[0]
    event = [verb for verb in verbs if verb.split("|")[5]=="VERB"]
    if event:
      return event[0]
    event = [verb for verb in verbs if "VB" in verb.split("|")[6]]
    if event:
      return event[0]
    event = [verb for verb in verbs if "ROOT" and "NN" not in verb]
    if event:
      return event[0]
    return verbs[0]

def get_event_idx(sent,nlp)->int:
  doc = nlp(sent)
  verbs = get_verbs(doc)
  tokens = get_word_idx(sent)
  event = get_event(verbs)
  tokens = [[i for i in range(int(token.split("|")[1]),int(token.split("|")[2]))] for token in tokens if token.split("|")[0]!=''] # skip whitespaces
  event_idx = int(event.split("|")[3])
  return int([tokens.index(token) for token in tokens if event_idx in token][0])
  # return tokens.index(int(event))

def load_sct_story(line,nlp):
  if 'AnswerRightEnding' in line.keys():  # SCT dev
    true_end = line['RandomFifthSentenceQuiz1'] if int(line['AnswerRightEnding']) == 1 else line['RandomFifthSentenceQuiz2']
    false_end = line['RandomFifthSentenceQuiz2'] if true_end == line['RandomFifthSentenceQuiz1'] else line['RandomFifthSentenceQuiz1']
  else:                                   # SCT 2018 test
    print('SCT label column not specified...')
    exit()
    # true_end = line['RandomFifthSentenceQuiz1']
    # false_end = line['RandomFifthSentenceQuiz2']
  sentences = [line['InputSentence1'],line['InputSentence2'],line['InputSentence3'],line['InputSentence4'],true_end,false_end]
  all_event_idx = [get_event_idx(sent,nlp) for sent in sentences]
  line_true = [line['InputStoryid'],[{'sent':line['InputSentence1'],'event':all_event_idx[0]},{'sent':line['InputSentence2'],'event':all_event_idx[1]},{'sent':line['InputSentence3'],'event':all_event_idx[2]},{'sent':line['InputSentence4'],'event':all_event_idx[3]}],{'sent':true_end,'event':all_event_idx[4]},1]
  line_false = [line['InputStoryid'],[{'sent':line['InputSentence1'],'event':all_event_idx[0]},{'sent':line['InputSentence2'],'event':all_event_idx[1]},{'sent':line['InputSentence3'],'event':all_event_idx[2]},{'sent':line['InputSentence4'],'event':all_event_idx[3]}],{'sent':false_end,'event':all_event_idx[5]},0]
  return line_true,line_false

def load_cmcnc_story(line,nlp):
  sentences = [line['Sentence1'],line['Sentence2'],line['Sentence3'],line['Sentence4'],line['Sentence5'],line['RandomFinalSentence']]
  all_event_idx = [get_event_idx(sent,nlp) for sent in sentences]
  line_true = [line['StoryID'],[{'sent':sentences[0],'event':all_event_idx[0]},{'sent':sentences[1],'event':all_event_idx[1]},{'sent':sentences[2],'event':all_event_idx[2]},{'sent':sentences[3],'event':all_event_idx[3]}],{'sent':sentences[4],'event':all_event_idx[4]},1]  # sentence 1-4 as input, 5 as target, label=True
  line_false = [line['StoryID'],[{'sent':sentences[0],'event':all_event_idx[0]},{'sent':sentences[1],'event':all_event_idx[1]},{'sent':sentences[2],'event':all_event_idx[2]},{'sent':sentences[3],'event':all_event_idx[3]}],{'sent':sentences[5],'event':all_event_idx[5]},0] # sentence 1-4 as input, 6 as target, label=False
  return line_true,line_false

def load_cmcnc_event(line,nlp):
  input_sentences = line['input'].split('|')
  all_event_idx = [1]*len(input_sentences)
  input_lines = [{'sent': sent+'.', 'event': event} for sent, event in zip(input_sentences, all_event_idx)]
  true_target_line = {'sent': line['target']+'.', 'event': 1}
  line_true = [line['StoryID'],input_lines,true_target_line,1]
  # line_true = {'StoryID':line['StoryID'],'input':input_sentences,'target':line['target'],'label':1}
  # sample_true = SampleCMCNC(line_true)
  # samples.append(sample_true)
  neg_sentences = line['neg'].split('|')
  false_lines = [[line['StoryID'],input_lines,{'sent':sent,'event':1},0] for sent in neg_sentences]
  # neg_samples = [{'input':input_sentences,'target':sent,'label':0} for sent in neg_sentences]
  # for line_false in neg_samples:
  #   sample_false = SampleCMCNC(line_false)
  #   samples.append(sample_false)
  return line_true,false_lines

class Sentence():

  def __init__(self, line):
    self.text = line['sent']
    self.event = line['event']
    self.raw_text = self.detokenize_sentence(self.text)
    self.tokens = self.get_tokens(self.text)
  
  def detokenize_sentence(self,sent):
    return re.sub(r'[^\w\s]', '', sent)

  def get_tokens(self,text):
    return text.split(' ')

  def get_tup(self,tups):
    return tups.split(',')[0].split('|')
  
  def __str__(self):
    return f"Text: {self.text}\nTokens: {self.tokens}\nEvent: {self.event}"

class Sample():

  def __init__(self, line):
    self.id = line[0]
    self.input = [Sentence(sent) for sent in line[1]]
    self.input_sent = (' ').join([sent.text for sent in self.input])
    self.input_event_idx = self.get_input_event_idx(self.input)

    self.target = Sentence(line[2])
    # self.target_sent = self.target.text
    # self.target_event_idx = self.target.event

    self.label = line[-1]
    self.representation = dict()
  
  def get_input_event_idx(self,input_list):
    offset = 0
    input_event_idx = []
    for i in input_list:
      event_idx = i.event+offset
      input_event_idx.append(event_idx)
      offset+=len(i.tokens)
    return input_event_idx
  
  def __str__(self):
    return f"Input: {self.input_sent}\nInput event: {self.input_event_idx}\nTarget: {self.target.text}\nTarget event: {self.target.event}\nLabel: {self.label}"

'''
Create a Sample object from each line of the input file.
'''
def load_all_samples(src_path:str, args, spacy_model="en_core_web_sm")->list:
    nlp = spacy.load(spacy_model)
    samples = []
    if args.data_set.lower() == "sct":
        data = pd.read_csv(src_path)
        loader = load_sct_story
    elif args.data_set.lower() == "nct":
        data = pd.read_csv(src_path,sep='\t')
        loader = load_cmcnc_story
    elif args.data_set.lower() == "cmcnc":
        data = pd.read_csv(src_path,sep='\t')
        loader = load_cmcnc_event
    for idx,row in data.iterrows():
        line_true,line_false=loader(row,nlp)
        sample_true = Sample(line_true)
        samples.append(sample_true)
        if args.data_set.lower() == "cmcnc":
          for line in line_false:
            sample_false = Sample(line)
            samples.append(sample_false)
        else:
          sample_false = Sample(line_false)
          samples.append(sample_false)
    label_list = set(sample.label for sample in samples)
    return samples, list(label_list)

'''
SVM training and evaluation.
'''
def train_eval(X_train, y_train, X_test, y_test, prob=False):

    scaler = MinMaxScaler()
    
    scaled_X_train = scaler.fit_transform(X_train)  # Scaling must be applied both to training and evaluation set using the same scale

    clf = LinearSVC(max_iter=50000, dual=False)
    clf = CalibratedClassifierCV(clf) if prob else clf   # enables LinearSVC to output probabilities

    if X_test:
      scaled_X_test = scaler.transform(X_test)
      clf.fit(scaled_X_train, y_train)
      y_pred = clf.predict(scaled_X_test)
      y_pred_proba = clf.predict_proba(scaled_X_test) if prob else []
      return classification_report(y_test, y_pred, output_dict=True), y_pred_proba
    else:
      return cross_val_score(clf, scaled_X_train, y_train, cv=5), []