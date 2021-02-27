import pickle, os, warnings, random
from random import sample
from pathlib import Path

import spacy
from spacy.util import minibatch, compounding

from label_studio.ml import LabelStudioMLBase


class CustomNER(LabelStudioMLBase):

    def __init__(self, **kwargs):        
        super(CustomNER, self).__init__(**kwargs)

        assert len(self.parsed_label_config) == 1
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        print('Initializing Model')
        print('with Info')
        print(self.info)

        assert self.info['type'] == 'Labels'

        assert len(self.info['to_name']) == 1
        assert len(self.info['inputs']) == 1
        assert self.info['inputs'][0]['type'] == 'Text'

        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

        if not self.train_output:
            #Load plain model
            self.spacy_model = spacy.load('en_core_web_sm')
            self.labels = self.info['labels']
            print('Initialized with from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name = self.to_name, labels = self.labels
            ))
        else:
            #Load model from previously training results
            print('Loading Model file {model_path}'.format(model_path=self.train_output['model_file']))

            self.spacy_model = spacy.load(self.train_output['model_file'])
            print('Trained Labels')
            print(self.spacy_model.get_pipe('ner').labels)
            self.labels = self.spacy_model.get_pipe('ner').labels

            print('Loaded from Train Output from_name={from_name}, to_name={to_name}, labels={labels}'.format(
                from_name=self.from_name, to_name = self.to_name, labels = self.labels
            ))

    def predict(self, tasks, **kwargs):
        print('Starting Prediction for {n} tasks'.format(n=len(tasks)))
        input_texts = [task['data'][self.value] for task in tasks]

        processed_documents = list(self.spacy_model.pipe(input_texts))

        return_object = []

        for doc in processed_documents:
            spacy_results = []
            if len(doc.ents) == 0:
                spacy_results = [{
                    'from_name' : 'label',
                    'to_name'   : 'text',
                    'type'      : 'labels',
                    'value'     :  {}
                }]
            else:
                spacy_results = [{
                    'from_name': 'label', 
                    'to_name': 'text', 
                    'type': 'labels', 
                    'value': {
                        'start' : ent.start_char,
                        'end': ent.end_char,
                        'labels': [ent.label_],
                        'text': ent.text}} for ent in doc.ents]
            
            return_object.append({'result': spacy_results})

        print(return_object)
        
        return(return_object)


    def fit(self, completions, workdir=None, **kwargs):
        print('Starting Model Fit for documents')
        TRAIN_DATA = []

        for completion in completions:
            input_text = completion['data'][self.value]
            
            output_labels = [ 
                (label_obj['value']['start'], 
                label_obj['value']['end'],
                label_obj['value']['labels'][0])
                for label_obj in completion['completions'][-1]['result'] 
                if len(label_obj['value']) > 0]

            TRAIN_DATA.append((input_text, {'entities': output_labels}))
        
        trained = self.train_model(TRAIN_DATA, workdir)        
        
        print('Training Complete')

        train_output = {
            'labels' : self.spacy_model.get_pipe('ner').labels,
            'model_file' : workdir
        }

        print('Labels Learned')
        print(train_output['labels'])

        return train_output


    def train_model(self,TRAIN_DATA, output_dir, n_iter=100):
        print('Model Training Function')
        ner = self.spacy_model.get_pipe('ner')
        
        for _, annotations in TRAIN_DATA:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])

        print('Updated Entities in NER')

        # SPACY TRAINING LOOP

        pipe_exceptions = ['ner', 'trf_wordpiecer','trf_tok2vec']
        other_pipes = [pipe for pipe in self.spacy_model.pipe_names if pipe not in pipe_exceptions]

        #Only Training NER
        print('Starting Training Loop')

        optimizer = self.spacy_model.resume_training()
        with self.spacy_model.disable_pipes(*other_pipes), warnings.catch_warnings():
            #Show warnings for misaligned entity spans only once
            warnings.filterwarnings('once', category=UserWarning,module='spacy')

            for itn in range(n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                batches = minibatch(TRAIN_DATA, size= compounding(4.0,32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.spacy_model.update(
                        texts, 
                        annotations,
                        drop=0.5,
                        sgd=optimizer,
                        losses = losses)
                print('\rProgress_{iteration}_{losses}'.format(iteration=itn, losses = losses), end='')
        
        for text, _ in sample(TRAIN_DATA,5):
            doc = self.spacy_model(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            self.spacy_model.to_disk(output_dir)
            print('Saved Model to {dir}'.format(dir=output_dir))

        return True


            

        
            
