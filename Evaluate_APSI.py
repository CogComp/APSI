from util import *
from sklearn.metrics.pairwise import cosine_similarity


def select_most_frequent_subevent(raw_schema):
    sorted_schema = sorted(raw_schema, key=lambda x: x['weight'], reverse=True)
    top_5_schema = sorted_schema[:5]
    event_by_temporal = sorted(top_5_schema, key=lambda x: x['raw_location'], reverse=True)
    return event_by_temporal


with open('intrinsic_dataset/process_structure_by_predicate.json', 'r') as f:
    process_structure_by_predicate = json.load(f)

with open('intrinsic_dataset/process_structure_by_argument.json', 'r') as f:
    process_structure_by_argument = json.load(f)

with open('intrinsic_dataset/test_processes.json', 'r') as f:
    test_processes = json.load(f)

test_process_to_structure = dict()

local_word_to_hypernym_path = dict()
all_w = list()
for tmp_predicate in process_structure_by_predicate:
    tmp_schema = process_structure_by_predicate[tmp_predicate]
    for tmp_e in tmp_schema:
        for tmp_w in tmp_e['name'].split('$$')[1:]:
            all_w.append(tmp_w.split(':')[1].split('.')[0])
for tmp_argument in process_structure_by_argument:
    tmp_schema = process_structure_by_argument[tmp_argument]
    for tmp_e in tmp_schema:
        for tmp_w in tmp_e['name'].split('$$')[1:]:
            all_w.append(tmp_w.split(':')[1].split('.')[0])
all_w = list(set(all_w))
for tmp_w in all_w:
    local_word_to_hypernym_path[tmp_w] = get_hypernym_path(tmp_w, 0.5, 0.5)

test_process_to_structure = dict()
merging_method = 'main'
process_count = dict()
for tmp_example in test_processes:
    tmp_process_key = tmp_example['process_key']
    if tmp_process_key not in process_count:
        process_count[tmp_process_key] = dict()
    if str(len(tmp_example['subevent_structures'])) not in process_count[tmp_process_key]:
        process_count[tmp_process_key][str(len(tmp_example['subevent_structures']))] = 0
    process_count[tmp_process_key][str(len(tmp_example['subevent_structures']))] += 1
all_test_processes = list()
for tmp_process_key in process_count:
    sorted_number_of_subevents = sorted(process_count[tmp_process_key],
                                        key=lambda x: process_count[tmp_process_key][x], reverse=True)
    all_test_processes.append((tmp_process_key, sorted_number_of_subevents[0]))

for tmp_process_pair in tqdm(all_test_processes):
    tmp_process_key = tmp_process_pair[0]
    tmp_length = int(tmp_process_pair[1])
    tmp_p = tmp_process_key
    tmp_predicate = tmp_process_key.split(' ')[0]
    tmp_argument = tmp_process_key.split(' ')[1]
    if tmp_predicate in process_structure_by_predicate:
        predicate_schema = process_structure_by_predicate[tmp_predicate]
    else:
        predicate_schema = list()
    if tmp_argument in process_structure_by_argument:
        argument_schema = process_structure_by_argument[tmp_argument]
    else:
        argument_schema = list()
    test_process_to_structure[tmp_p] = predict_structure(local_word_to_hypernym_path, predicate_schema, argument_schema,
                                                         tmp_length, 0.5, 0.5, merging_method)

with open('intrinsic_dataset/APSI_prediction.json', 'w') as f:
    json.dump(test_process_to_structure, f)

with open('intrinsic_dataset/APSI_prediction.json', 'r') as f:
    test_process_to_structure = json.load(f)

evaluate_multi_reference(test_processes, test_process_to_structure)

print('end')
