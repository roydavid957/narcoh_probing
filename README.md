# Probing Language Models for Narrative Coherence

## Code
- preprocess.py: preprocesses the data to extract relevant data. Usage: "python3 preprocess.py --file_name FILE_PATH --data_set {story, event} --use_prot {True, False} --output_file OUTPUT_PATH"
- probing.py: script for per layer probing of PTLMs using a SVM. Usage: "python3 probing.py --model_ckpt MODEL_CHECKPOINT --train_file TRAIN_PATH --test_file TEST_PATH --data_set {SCT, NCT, CMCNC} --output_file OUTPUT_PATH --output_dir OUTPUT_DIR_PATH --output_prob {True, False} --use_event_embeddings {True, False} --use_sentence_embeddings {True, False}"

## Datasets
- Story Cloze Task (SCT): 5 sentence short stories (https://www.cs.rochester.edu/nlp/rocstories/).
- Narrative Cloze Task (NCT): 5 sentence stories extracted from English Gigaword corpus.
- Coherent Multiple Choice Narrative Cloze (CMCNC): event triplets (https://github.com/StonyBrookNLP/event-tensors/tree/master/data).
