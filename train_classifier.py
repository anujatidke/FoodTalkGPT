# Code for adapted a transformer-based language model
# Author: Jordan Boyd-Graber
# Date: 26. Sept 2022

import argparse
from collections import Counter
from random import sample

import numpy as np

import pyarrow as pa

from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
#import datasets
from transformers import DataCollatorWithPadding



kUNK = "~UNK~"

def accuracy(eval_pred):
    """
    Compute accuracy for the classification task using the
    load_metric function.  This function is needed for the
    compute_metrics argument of the Trainer.

    You shouldn't need to modify this function.

    Keyword args:
    eval_pred -- Output from a classifier with the logits and labels.
    """

    metric = load_metric("accuracy")
    logits, label = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=label)


class DatasetTrainer:
    def __init__(self, checkpoint: str='distilbert-base-cased', train_set: list[str]=['guesstrain']):
        """
        Initialize a class to train a fine-tuned BERT model.

        Args:
          checkpoint - model we build off of
          train_set - a list of the folds we use to train the model
        """
        self._checkpoint = checkpoint
        self._train = None
        self._train_fold_name = train_set
        self._model = AutoTokenizer.from_pretrained(self._checkpoint)
        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)


    def load_qb_data(self, desired_label: str='category', max_labels: int=5000, min_frequency: int=3, limit: int=-1):
        """
        Load the QANTA dataset and convert one of its columns into a label we can use for predictions.

        Args:
          desired_label - The column in the dataset to use as our classification label
          max_labels - How many labels total we can have at most
          min_frequency - How many times a label must appear to be counted
          limit - How many examples we have per fold
        """
        self._dataset = load_dataset("qanta", 'mode=full,char_skip=25')
        self._dataset = self._dataset['guesstrain']
        # Let's limit the number of examples we have in each fold
        self._dataset = self._dataset.select(range(limit))

        # Build the label set
        label_counts = Counter(self._dataset[desired_label])
        self._labels = [x for x, y in label_counts.most_common(max_labels) if y >= min_frequency]
        self._label_map = {x: i for i, x in enumerate(self._labels)}


        # Map the column into your newly-defined label set
        def map_label(dataset):
            dataset["labels"] = self._label_map.get(dataset[desired_label], -1)
            return dataset

        self._dataset = self._dataset.map(map_label)


        '''
        ######################################################
        label = self._dataset["guesstrain"]["category"]
        #print(labels)

        # Build the label set
        label_counts = Counter(label)
        unique_labels = [label for label, count in label_counts.items() if count >= min_frequency]

        # Limit the number of unique labels
        unique_labels = unique_labels[:max_labels]

        # Map the column into the newly-defined label set
        label_mapping = {label: i for i, label in enumerate(unique_labels)}

        # Turn labels into class encodings
        class_encodings = [label_mapping[label] for label in label]
        
        #print(class_encodings)
        ############

        # TODO: Modify the dataset so that you can predict with the
        # column you want on a subset of the data.
        
        # Build the label set


        # Map the column into your newly-defined label set
        # And turn that into a class encoding'''


    def tokenize_data(self):
        """
        Tokenize our data so that it's ready for BERT.

        You should not need to modify this function.
        """
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        tokenize_function = lambda x: tokenizer(x["full_question"], padding="max_length", truncation=True)
        self._tokenized = self._dataset.map(tokenize_function, batched=True)
        return self._tokenized

    def load_and_train_model(self, epochs: int=2):
        """
        Load a BERT sequence classifier and then fine-tune it on our date.

        Args:
          epochs - How many epochs to train.
        """
        # TODO: We have provided code to load the model, but you need to actually train the model, which is not implemented!
        from transformers import AutoModelForSequenceClassification

        # TODO: Get rid of this magic number for the number of categories
        num_categories = 11

        print("Using %i categories" % num_categories)

        model = AutoModelForSequenceClassification.from_pretrained(self._checkpoint, num_labels=num_categories)
        split_data = self._tokenized.train_test_split(test_size=0.1)
        train_split , eval_split = split_data['train'], split_data['test']

        training_args = TrainingArguments(
            output_dir= "./result",
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            evaluation_strategy="steps",
            save_total_limit=2,
            eval_steps=500,  
            save_steps=500,  
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir="./logs"
        )
        data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_split,
            eval_dataset=eval_split,
            tokenizer=self._tokenizer,
            compute_metrics= accuracy,
            data_collator=data_collator
        )
        trainer.train()

        # Evaluate the model on the validation set
        evaluation_results = trainer.evaluate()

        print("Evaluation Results:", evaluation_results)

 



        return model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--limit", help="Limit folds to this many examples",
                           type=int, default=10, required=False)
    argparser.add_argument("--to_classify", help="Target field that model will try to predict",
                           type=str, default="category")
    argparser.add_argument("--min_frequency", help="How many times must a label appear to be predicted",
                           type=int, default=5)
    argparser.add_argument("--max_label", help="How many labels (maximum) will we predict",
                           type=int, default=5000)    
    argparser.add_argument("--ec", help="Extra credit option (df, lazy, or rate)",
                           type=str, default="")

    flags = argparser.parse_args()

    dt = DatasetTrainer()
    
    dt.load_qb_data(desired_label=flags.to_classify,
                    max_labels=flags.max_label,
                    min_frequency=flags.min_frequency,
                    limit=flags.limit)
    dt.tokenize_data()
    model = dt.load_and_train_model()
    model.save_pretrained("finetuned.model")
    
    
