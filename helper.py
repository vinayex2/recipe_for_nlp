import os, argparse
import spacy
import inflection
import hashlib
import json 

# Initiate Label Studio and create project setup
def process(file_name,model):
    print('Processing file {0}'.format(file_name))
    nlp = spacy.load(model)
    hashes = []
    sentences = []
    with open(file_name,'r') as f:
        for line in f:
            try:
                '''docner = nlp(inflection.transliterate(line.replace('\n',' ')))
                labels = []                

                for ent in docner.ents:
                    if (str(ent.label_) != 'IGNORE'):
                        labels.append([ent.start_char-ent.sent.start_char, ent.end_char-ent.sent.start_char,ent.label_])
                        #labels.append([ent.start_char, ent.end_char,ent.label_])

                text = str(docner)'''
                text = line.replace('\n',' ')
                result = hashlib.sha224(text.encode()).hexdigest()
                #print(labels)

                if result not in hashes:
                    if len(text.strip()) > 10:
                        sentences.append({'text': text})
                    hashes.append(result)

            except Exception as e:
                print(e)
                break
    
    with open('task.json', 'w', encoding='utf-8') as f:
        f.write('[')
        for sentence in sentences:
            f.write(json.dumps(sentence))
            f.write(',')
        f.write('{"text": "DUMMY"}]')

    return True           

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper Script for Create Tasks for Label Studio Import (preserves Base Spacy Entities)')
    parser.add_argument('mode', choices=['create_tasks'], help='Tasks Json file for import')
    parser.add_argument('--data', help='Raw File Path', default='sample_data\\famous_quotes.txt')
    parser.add_argument('--model', default='en_core_web_sm', help = 'Spacy model to use (defaults to en_core_web_sm). Can use custom NER as input')

    args = parser.parse_args()

    if args.mode =='create_tasks':
        process(args.data,args.model)