import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('job', type=str)
args = parser.parse_args()

meaning_prompt = "Which of these is more similar in meaning to the original text? Enter '1' or '2', or '0' if they are equally similar.\n"
grammar_prompt = "Which of these is most grammatically and syntactically correct? Enter '1' or '2', or '0' if they are equally correct.\n"
classification_prompt = "What category does this text most likely correspond to? '0' for Business, '1' for World, '2' for Sports, '3' for Sci/Tech.\n"

# old = single sub
# final = multi sub
single_path_slim = 'ag_news_old3/ag_news_old3_slim.txt'
multi_path_slim = 'ag_news_final3/ag_news_final3_slim.txt'
single_path = 'ag_news_old3/ag_news_old3.csv'
multi_path = 'ag_news_final3/ag_news_final3.csv'

single_dict = {}
single_dict['original_text'] = []
single_dict['perturbed_text'] = []
single_dict['result_type'] = []
single_dict['original_class'] = []

with open(single_path_slim, 'r') as f:
    index = f.readline().strip()
    while index.isnumeric():
        status = f.readline().strip()
        single_dict['original_text'].append(f.readline().strip())
        single_dict['perturbed_text'].append(f.readline().strip())
        if "FAILED" not in status and "SKIPPED" not in status:
            single_dict['result_type'].append("Successful")
        else:
            single_dict['result_type'].append("Failed") 
        single_dict['original_class'].append(status.split(" ")[0][2:])
        f.readline()
        index = f.readline()[:-1]
        
multi_dict = {}
multi_dict['original_text'] = []
multi_dict['perturbed_text'] = []
multi_dict['result_type'] = []
multi_dict['original_class'] = []

with open(multi_path_slim, 'r') as f:
    index = f.readline().strip()
    while index.isnumeric():
        status = f.readline().strip()
        multi_dict['original_text'].append(f.readline().strip())
        multi_dict['perturbed_text'].append(f.readline().strip())
        if "FAILED" not in status and "SKIPPED" not in status:
            multi_dict['result_type'].append("Successful")
        else:
            multi_dict['result_type'].append("Failed") 
        multi_dict['original_class'].append(status.split(" ")[0][2:])
        
        f.readline()
        index = f.readline()[:-1]
        
single = pd.DataFrame(single_dict)
multi = pd.DataFrame(multi_dict)

both_succesful = (single['result_type'] == 'Successful') & (multi['result_type'] == 'Successful')

single = single[both_succesful]
multi = multi[both_succesful]

single.reset_index(inplace=True)
multi.reset_index(inplace=True)

np.random.seed(42)
sample_idxs = np.random.choice(single['index'], size=50)

single_sample = single[single['index'].apply(lambda x: x in sample_idxs)]
multi_sample = multi[multi['index'].apply(lambda x: x in sample_idxs)]

assert all(single_sample['original_text'] == multi_sample['original_text'])

single_sample.set_index('index', inplace=True)
multi_sample.set_index('index', inplace=True)

if args.job == 'single':

    try:
        single_classification_results = pd.read_csv('single_classification_results_ag_news.csv')
    except:
        single_classification_results = pd.DataFrame(columns=['index', 'perturbed_text', 'original_class', 'user_classification'])

    user_in = ""
    already_done = set(single_classification_results['index'])
    print('Enter "q" to quit and save your results so far.')

    while user_in != 'q' and len(set(sample_idxs) - set(already_done)) > 0:
        next_index = np.random.choice(list(set(sample_idxs) - set(already_done)))

        single_perturbed = single_sample['perturbed_text'][next_index]
        single_class = single_sample['original_class'][next_index]

        print("Perturbed Text: ", single_perturbed, sep='\n\n')
        decode = {'0':'Business', '1':'World', '2':'Sports', '3': 'Sci/Tech'}

        user_in = input(classification_prompt)
        while user_in not in ['0', '1', '2', '3', 'q']:
            user_in = input('Sorry, not a valid response, try again: ')
        if user_in == 'q':
            break
        classification = decode[user_in]

        row = {
            'index': next_index,
            'perturbed_text': single_perturbed,
            'original_class': single_class,
            'user_classification': classification
        }
        single_classification_results = single_classification_results.append(row, ignore_index=True)
        already_done = set(single_classification_results['index'])

    single_classification_results.to_csv('single_classification_results_ag_news.csv')
    
elif args.job == 'multi':
    
    try: 
        multi_classification_results = pd.read_csv('multi_classification_results_ag_news.csv')
    except:
        multi_classification_results = pd.DataFrame(columns=['index', 'perturbed_text', 'original_class', 'user_classification'])

    user_in = ""
    already_done = set(multi_classification_results['index'])
    print('Enter "q" to quit and save your results so far.')

    while user_in != 'q' and len(set(sample_idxs) - set(already_done)) > 0:
        next_index = np.random.choice(list(set(sample_idxs) - set(already_done)))

        multi_perturbed = multi_sample['perturbed_text'][next_index]
        multi_class = multi_sample['original_class'][next_index]

        print("Perturbed Text: ", multi_perturbed, sep='\n\n')
        decode = {'0':'Business', '1':'World', '2':'Sports', '3': 'Sci/Tech'}

        user_in = input(classification_prompt)
        while user_in not in ['0', '1', '2', '3', 'q']:
            user_in = input('Sorry, not a valid response, try again: ')
        if user_in == 'q':
            break
        classification = decode[user_in]

        row = {
            'index': next_index,
            'perturbed_text': multi_perturbed,
            'original_class': multi_class,
            'user_classification': classification
        }
        multi_classification_results = multi_classification_results.append(row, ignore_index=True)
        already_done = set(multi_classification_results['index'])


    multi_classification_results.to_csv('multi_classification_results_ag_news.csv')
    
elif args.job == 'compare':
    
    try: 
        ranking_results = pd.read_csv('ranking_results_ag_news.csv')
    except:
        ranking_results = pd.DataFrame(columns=['index', 'original_text', 'original_class', 'multi_perturbed_text', 'single_perturbed_text', 'preferred', 'grammatical'])

    user_in = ""
    already_done = set(ranking_results['index'])
    print('Enter "q" to quit and save your results so far.')

    while user_in != "q" and len(set(sample_idxs) - set(already_done)) > 0:
        next_index = np.random.choice(list(set(sample_idxs) - set(already_done)))
        assert single_sample['original_text'][next_index] == multi_sample['original_text'][next_index]

        original = single_sample['original_text'][next_index]
        single_perturbed = single_sample['perturbed_text'][next_index]
        multi_perturbed = multi_sample['perturbed_text'][next_index]

        # Switch up the order at random so you cannot rely on the positioning to determine which is which
        if np.random.random() > 0.5:
            print("Original Text:", original, "Pertubed Text 1:", single_perturbed, "Perturbed Text 2:", multi_perturbed, sep='\n\n')
            decode = {'0':'nuetral', '1':'single', '2':'multi'}
        else:
            print("Original Text:", original, "Perturbed Text 1:", multi_perturbed, "Pertubed Text 2:", single_perturbed, sep='\n\n')
            decode = {'0':'nuetral', '1':'multi', '2':'single'}
        # Meaning
        user_in = input(meaning_prompt)
        while user_in not in ['0', '1', '2', 'q']:
            user_in = input('Sorry, not a valid response, try again: ')
        if user_in == 'q':
            break
        preferred = decode[user_in]
        # Grammar  
        user_in = input(grammar_prompt) 
        while user_in not in ['0', '1', '2', 'q']:
            user_in = input('Sorry, not a valid response, try again: ')
        if user_in == 'q':
            break
        grammar = decode[user_in]
        row = {
            'index' : next_index,
            'original_text' : original,
            'original_class' : single_sample['original_class'][next_index],
            'multi_perturbed_text': multi_perturbed,
            'single_perturbed_text': single_perturbed,
            'preferred': preferred,
            'grammatical': grammar
        }
        ranking_results = ranking_results.append(row, ignore_index=True)
        already_done = set(ranking_results['index'])
    ranking_results.to_csv('ranking_results_ag_news.csv')
    


else:
    print('Job specified is not valid. Either "compare", "single", or "multi"')



