from util import *
from multiprocessing import Pool


class DataLoader(object):
    def __init__(self, data_path):
        self.processes = list()
        with open(data_path, 'r') as data_f:
            raw_processes = json.load(data_f)
            for tmp_raw_process in raw_processes:
                self.processes.append(Process(tmp_raw_process))
        self.id2loc = dict()
        for i, tmp_process in enumerate(self.processes):
            self.id2loc[tmp_process.id] = i


def processes_to_schemas(predicate_processes, argument_processes, weights):
    verb_weight = weights[0]
    noun_weight = weights[1]
    process_structure_by_predicate = dict()
    process_structure_by_argument = dict()
    print('verb weight:', verb_weight, 'noun weight:', noun_weight)
    for tmp_predicate in tqdm(predicate_processes):
        if len(predicate_processes[tmp_predicate]) > 0:
            process_structure_by_predicate[tmp_predicate] = schema_induction(predicate_processes[tmp_predicate][:200], verb_weight, noun_weight)
        else:
            process_structure_by_predicate[tmp_predicate] = list()

    for tmp_argument in tqdm(argument_processes):
        if len(argument_processes[tmp_argument]) > 0:
            process_structure_by_argument[tmp_argument] = schema_induction(argument_processes[tmp_argument][:200], verb_weight, noun_weight)
        else:
            process_structure_by_argument[tmp_argument] = list()

    with open('intrinsic_dataset/process_structure_by_predicate.json', 'w') as f:
        json.dump(process_structure_by_predicate, f)

    with open('intrinsic_dataset/process_structure_by_argument.json', 'w') as f:
        json.dump(process_structure_by_argument, f)



with open('intrinsic_dataset/train_processes.json', 'r') as f:
    raw_train_data = json.load(f)

with open('intrinsic_dataset/test_processes.json', 'r') as f:
    raw_test_data = json.load(f)


test_predicates = list()
test_arguments = list()
for tmp_test_process in tqdm(raw_test_data):
    tmp_predicate = tmp_test_process['process_predicate']
    tmp_argument = tmp_test_process['process_argument']
    test_predicates.append(tmp_predicate)
    test_arguments.append(tmp_argument)

test_predicates = list(set(test_predicates))
test_arguments = list(set(test_arguments))

predicate_processes = dict()
argument_processes = dict()
for tmp_predicate in test_predicates:
    predicate_processes[tmp_predicate] = list()

for tmp_argument in test_arguments:
    argument_processes[tmp_argument] = list()

for tmp_train_process in raw_train_data:
    tmp_train_predicate = tmp_train_process['process_predicate']
    tmp_train_argument = tmp_train_process['process_argument']
    if tmp_train_predicate in predicate_processes:
        predicate_processes[tmp_train_predicate].append(Process(tmp_train_process).subevents_aser)
    if tmp_train_argument in argument_processes:
        argument_processes[tmp_train_argument].append(Process(tmp_train_process).subevents_aser)

processes_to_schemas(predicate_processes, argument_processes, (0.5, 0.5))

print('end')
