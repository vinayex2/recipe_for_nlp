This repository contains working code to start NLP projects with Human in the loop.

Imagine you have a collection of unlabeled documents and are asked to extract information using NLP model

Using this code, you can get from raw unlabeled documents to NLP Model in less than a day's work
You can use this template and replace sample documents with your own dataset to quickly get started.

Current Setup works for Named Entity Recoginition Use case. However, it can be modified to handle other use cases.

Requires
1. Python version 3.8.5
2. Spacy Version 2.3.2
3. Label Studio
 -  Supports variety of NLP annotation tasks

Works with Python 3.8.5

Steps:
1. Modify labeling_config.xml to add your custom labels
2. Modify sample data to include your own dataset
3. Run python helper create_tasks dataset_file_path to generate tasks for Label Studio
3. Use File Commands.txt to get started with setup for Label Studio


The setup will 
1. Configure Label Studio for your annotation task
2. Set up a ML Backend using Spacy to automatically train on your annotations (can plugin other ML backends as well)

Use https://labelstud.io/ for user guide and examples

