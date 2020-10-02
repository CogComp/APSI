# import torch
# from pytorch_transformers import *
import logging
import argparse
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn.functional as F
import os
import math
import ujson as json
from tqdm import tqdm
import random
# from allennlp.predictors.predictor import Predictor
# from allennlp.models.archival import load_archive
# from contextlib import ExitStack
import math
from functools import partial
# from multiprocessing import Pool
import spacy
import re
import time
import pickle
from wn import WordNet
import numpy as np
import collections


# from pycorenlp import StanfordCoreNLP

class Activity:
    def __init__(self, initial_relations):
        if initial_relations is None:
            self.parsed_relations = list()
            self.skeleton_parsed_relations = list()
            self.skeleton_words = list()
            self.words = list()
        else:
            self.parsed_relations = initial_relations['parsed_relations']
            self.skeleton_parsed_relations = initial_relations['skeleton_parsed_relations']
            self.skeleton_words = initial_relations['skeleton_words']
            self.words = initial_relations['words']
        self.pattern = 'NA'

    def update_pattern(self, pattern):
        self.pattern = pattern

    def remove_one_edge(self, edge):
        new_parsed_relations = list()
        new_skeleton_parsed_relations = list()
        for old_edge in self.parsed_relations:
            if old_edge[0][0] == edge[0][0] and old_edge[2][0] == edge[2][0]:
                continue
            new_parsed_relations.append(old_edge)
        for old_edge in self.skeleton_parsed_relations:
            if old_edge[0][0] == edge[0][0] and old_edge[2][0] == edge[2][0]:
                continue
            new_skeleton_parsed_relations.append(old_edge)
        self.parsed_relations = new_parsed_relations
        self.skeleton_parsed_relations = new_skeleton_parsed_relations
        self.find_skeleton_words()

    def find_skeleton_words(self):
        all_skeleton_words = list()
        for relation in self.skeleton_parsed_relations:
            if relation[0] not in all_skeleton_words:
                all_skeleton_words.append(relation[0])
            if relation[2] not in all_skeleton_words:
                all_skeleton_words.append(relation[2])
        self.skeleton_words = sorted(all_skeleton_words, key=lambda tup: tup[0])

        all_words = list()
        for relation in self.parsed_relations:
            if relation[0] not in all_words:
                all_words.append(relation[0])
            if relation[2] not in all_words:
                all_words.append(relation[2])
        self.words = sorted(all_words, key=lambda tup: tup[0])

    def contain_word(self, w):
        for relation in self.parsed_relations:
            if relation[0][1] == w or relation[2][1] == w:
                return True
        return False

    def contain_tuple(self, t):
        for relation in self.parsed_relations:
            if relation[0] == t or relation[2] == t:
                return True
        return False

    def to_string(self):
        all_words = list()
        for relation in self.parsed_relations:
            if relation[0] not in all_words:
                all_words.append(relation[0])
            if relation[2] not in all_words:
                all_words.append(relation[2])
        sorted_words = sorted(all_words, key=lambda tup: tup[0])
        generated_string = ''
        for tmp_word in sorted_words:
            generated_string += ' '
            generated_string += tmp_word[1]
        return generated_string[1:]

    def to_unlemmatize_string(self, original_sentence):
        all_words = list()
        for relation in self.parsed_relations:
            if relation[0] not in all_words:
                all_words.append(relation[0])
            if relation[2] not in all_words:
                all_words.append(relation[2])
        sorted_words = sorted(all_words, key=lambda tup: tup[0])
        generated_string = ''
        for tmp_word in sorted_words:
            generated_string += ' '
            generated_string += original_sentence['tokens'][tmp_word[0] - 1]
        return generated_string[1:]

    def to_dict(self, original_sentence=None):
        if original_sentence:
            original_skeleton_tokens = list()
            original_tokens = list()
            for w_tuple in self.skeleton_words:
                if 'VB' in w_tuple[2]:
                    original_skeleton_tokens.append(original_sentence['tokens'][w_tuple[0] - 1])
                else:
                    original_skeleton_tokens.append(w_tuple[1])
            for w_tuple in self.words:
                if 'VB' in w_tuple[2]:
                    original_tokens.append(original_sentence['tokens'][w_tuple[0] - 1])
                else:
                    original_tokens.append(w_tuple)
        else:
            original_skeleton_tokens = list()
            original_tokens = list()
            for w_tuple in self.skeleton_words:
                if 'VB' in w_tuple[2]:
                    original_skeleton_tokens.append(w_tuple[1])
            for w_tuple in self.words:
                original_tokens.append(w_tuple)
        return {'parsed_relations': self.parsed_relations, 'skeleton_parsed_relations': self.skeleton_parsed_relations,
                'skeleton_words': self.skeleton_words, 'words': self.words, 'skeleton_tokens': original_skeleton_tokens,
                'tokens': original_tokens}

    def get_average_position(self):
        positions = list()
        for t in self.words:
            positions.append(t[0])
        try:
            average_1_position = sum(positions) / len(positions)
        except:
            print(self.to_dict())
            average_1_position = sum(positions) / len(positions)
        return average_1_position


class Rule:
    def __init__(self, rules):
        if rules is None:
            self.positive_rules = list()
            self.negative_rules = list()
        else:
            self.positive_rules = rules['positive_rules']
            self.negative_rules = rules['negative_rules']


class Activity_Rule:
    def __init__(self):
        self.positive_rules = list()
        self.possible_rules = list()
        self.negative_rules = list()


class ProbaseConcept(object):
    """
        Copied from https://github.com/ScarletPan/probase-concept
    """

    def __init__(self, data_concept_path=None):
        self.concept2idx = dict()
        self.idx2concept = dict()
        self.concept_inverted_list = dict()
        self.instance2idx = dict()
        self.idx2instance = dict()
        self.instance_inverted_list = dict()
        if data_concept_path:
            self._load_raw_data(data_concept_path)

    def _load_raw_data(self, data_concept_path):
        st = time.time()
        print("[probase-concept] Loading Probase files...")
        with open(data_concept_path) as f:
            triple_lines = [line.strip() for line in f]

        total_count = 0
        print("[probase-concept] Building index...")
        for line in tqdm(triple_lines):
            concept, instance, freq = line.split('\t')
            if concept not in self.concept2idx:
                self.concept2idx[concept] = len(self.concept2idx)
            concept_idx = self.concept2idx[concept]
            if instance not in self.instance2idx:
                self.instance2idx[instance] = len(self.instance2idx)
            instance_idx = self.instance2idx[instance]
            if concept_idx not in self.concept_inverted_list:
                self.concept_inverted_list[concept_idx] = list()
            self.concept_inverted_list[concept_idx].append((instance_idx, int(freq)))
            if instance_idx not in self.instance_inverted_list:
                self.instance_inverted_list[instance_idx] = list()
            self.instance_inverted_list[instance_idx].append((concept_idx, int(freq)))
            total_count += int(freq)

        self.N = total_count
        self.idx2concept = {val: key for key, val in self.concept2idx.items()}
        self.idx2instance = {val: key for key, val in self.instance2idx.items()}
        print("[probase-concept] Loading data finished in {:.2f} s".format(time.time() - st))

    def conceptualize(self, instance, score_method="likelihood"):
        """ Conceptualize given instance
        :type instance: str
        :type score_method: str
        :param instance: input instance such as "microsoft"
        :param score_method: "likelihood" or "pmi"
        :return:
        """
        if instance not in self.instance2idx:
            return []
        instance_idx = self.instance2idx[instance]
        instance_freq = self.get_instance_freq(instance_idx)
        concept_list = self.instance_inverted_list[instance_idx]
        rst_list = list()
        for concept_idx, co_occurrence in concept_list:
            if score_method == "pmi":
                score = np.log(self.N * co_occurrence / \
                               self.get_concept_freq(concept_idx) / \
                               instance_freq)
            elif score_method == "likelihood":
                score = co_occurrence / instance_freq
            else:
                raise NotImplementedError
            rst_list.append((self.idx2concept[concept_idx], score))
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def instantiate(self, concept):
        """ Retrieve all instances of a concept
        :type concept: str
        :param concept: input concept such as "company"
        :return:
        """
        if concept not in self.concept2idx:
            return []
        concept_idx = self.concept2idx[concept]
        rst_list = [(self.idx2instance[idx], freq) for idx, freq
                    in self.concept_inverted_list[concept_idx]]
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def get_concept_chain(self, instance, max_chain_length=5):
        if instance in self.concept2idx:
            chain = [instance]
        else:
            chain = list()
        tmp_instance = instance
        while True:
            concepts = self.conceptualize(tmp_instance, score_method="likelihood")
            if concepts:
                chain.append(concepts[0][0])
            else:
                break
            if len(chain) >= max_chain_length:
                break
            tmp_instance = chain[-1]
        if chain and chain[0] != instance:
            return [instance] + chain
        else:
            return chain

    def get_concept_freq(self, concept):
        if isinstance(concept, str):
            if concept not in self.concept2idx:
                return 0
            concept = self.concept2idx[concept]
        elif isinstance(concept, int):
            if concept not in self.idx2concept:
                return 0
        return sum([t[1] for t in self.concept_inverted_list[concept]])

    def get_instance_freq(self, instance):
        if isinstance(instance, str):
            if instance not in self.instance2idx:
                return 0
            instance = self.instance2idx[instance]
        elif isinstance(instance, int):
            if instance not in self.idx2instance:
                return 0
        return sum([t[1] for t in self.instance_inverted_list[instance]])

    def save(self, saved_path):
        st = time.time()
        print("[probase-concept] Loading data to {}".format(saved_path))
        with open(saved_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print("[probase-concept] Saving data finished in {:.2f} s".format(time.time() - st))

    def load(self, load_path):
        st = time.time()
        print("[probase-concept] Loading data from {}".format(load_path))
        with open(load_path, "rb") as f:
            tmp_dict = pickle.load(f)
        for key, val in tmp_dict.items():
            self.__setattr__(key, val)
        print("[probase-concept] Loading data finished in {:.2f} s".format(time.time() - st))

    @property
    def concept_size(self):
        return len(self.concept2idx)

    @property
    def instance_size(self):
        return len(self.instance2idx)


class SeedConcept(object):
    def __init__(self):
        self.person = "__PERSON__"
        self.url = "__URL__"
        self.digit = "__DIGIT__"
        self.year = "__YEAR__"
        self.person_pronoun_set = frozenset(
            ["he", "she", "i", "him", "her", "me", "woman", "man", "boy", "girl", "you", "we", "they"])
        self.pronouns = self.person_pronoun_set | frozenset(['it'])
        self.url_pattern = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'

    def check_is_person(self, word):
        return word in self.person_pronoun_set

    def check_is_year(self, word):
        if not word.isdigit() or len(word) != 4:
            return False
        d = int(word)
        return 1600 <= d <= 2100

    def check_is_digit(self, word):
        return word.isdigit()

    def check_is_url(self, word):
        if re.match(self.url_pattern, word):
            return True
        else:
            return False

    def is_seed_concept(self, word):
        return word in self.__dict__.values()

    def is_pronoun(self, word):
        return word in self.pronouns


class Node(object):
    def __init__(self, token, event_position, node_type, local_id):
        self.token = token
        self.event_position = event_position
        self.node_type = node_type
        self.local_id = local_id


class Edge(object):
    def __init__(self, start_node, edge_type, end_node):
        self.start_node = start_node
        self.edge_type = edge_type
        self.end_node = end_node


class Process(object):
    def __init__(self, input_process):
        self.raw_data = input_process
        self.id = input_process['id']
        self.number_of_subevents = len(input_process['subevents'])
        self.process_name = input_process['title']
        self.raw_process = input_process
        self.raw_subevents = input_process['subevents']
        self.subevents_aser = list()
        for tmp_raw_event in input_process['subevents_activities']:
            tmp_event = Activity(tmp_raw_event[0][1])
            tmp_event.update_pattern(tmp_raw_event[0][0])
            self.subevents_aser.append(tmp_event)
        self.nodes = list()
        self.edges = list()
        # last_predicate = None
        for i, tmp_subevent in enumerate(input_process['subevent_v_os']):
            tmp_predicate = tmp_subevent['V']
            tmp_argument = tmp_subevent['ARG1']
            tmp_predicate_node = Node(token=tmp_predicate, event_position=i, node_type='V', local_id=len(self.nodes))
            self.nodes.append(tmp_predicate_node)
            tmp_argument_node = Node(token=tmp_argument, event_position=i, node_type='A', local_id=len(self.nodes))
            self.nodes.append(tmp_argument_node)
            self.edges.append(Edge(start_node=tmp_predicate_node, edge_type='ARG1', end_node=tmp_argument_node))
            # if last_predicate:
            #     self.edges.append(Edge(start_node=last_predicate, edge_type='Temporal', end_node=tmp_predicate_node))
            last_predicate = tmp_predicate_node

    def show_process_name(self):
        print('Process name:', self.process_name)

    def show_subevents_v_o(self):
        output = ''
        for i in range(self.number_of_subevents):
            for tmp_edge in self.edges:
                if tmp_edge.edge_type == 'ARG1':
                    if tmp_edge.start_node.event_position == i:
                        output += '(' + tmp_edge.start_node.token + '->' + tmp_edge.end_node.token + ')'
                        if i != self.number_of_subevents - 1:
                            output += '-->'
        print('verb + arg1:', output)

    def show_subevents_sentence(self):
        output = ''
        for tmp_subevent in self.raw_subevents:
            output += '(' + tmp_subevent + ')'
            output += '-->'
        print('full subevents:', output[:-3].encode('utf-8'))

    def show_subevents_aser(self):
        output = ''
        for tmp_subevent in self.subevents_aser:
            output += '(' + tmp_subevent.to_string() + ')'
            output += '-->'
        print('our subevents:', output[:-3].encode('utf-8'))


def calculate_graph_overlap(edges_1, edges_2):
    number_of_match = 0
    number_of_token = 0
    for i, tmp_edge in enumerate(edges_1):
        if tmp_edge.start_node.token == edges_2[i].start_node.token:
            number_of_match += 1
        if tmp_edge.end_node.token == edges_2[i].end_node.token:
            number_of_match += 1
        number_of_token += 2
    return number_of_match / number_of_token


def sub_process_similarity_score(process_1, process_2):
    if process_1.number_of_subevents < process_2.number_of_subevents:
        span_length = process_1.number_of_subevents
        max_score = 0
        for i in range(process_2.number_of_subevents - process_1.number_of_subevents):
            tmp_score = calculate_graph_overlap(process_1.edges[:span_length], process_2.edges[i:i + span_length])
            if tmp_score > max_score:
                max_score = tmp_score
        return max_score
    elif process_1.number_of_subevents == process_2.number_of_subevents:
        return calculate_graph_overlap(process_1.edges, process_2.edges)
    else:
        span_length = process_2.number_of_subevents
        max_score = 0
        for i in range(process_1.number_of_subevents - process_2.number_of_subevents):
            tmp_score = calculate_graph_overlap(process_1.edges[i:i + span_length], process_2.edges[:span_length])
            if tmp_score > max_score:
                max_score = tmp_score
        return max_score


class DemoDataLoader:
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        self.processes = list()
        for tmp_raw_process in self.dataset:
            self.processes.append(Process(tmp_raw_process))
        self.id2loc = dict()
        for i, tmp_p in enumerate(self.processes):
            self.id2loc[tmp_p.id] = i


class IncompleteProcess(object):
    def __init__(self, process_id, process_name, first_part_events, second_part_events):
        self.id = process_id
        self.process_name = process_name
        self.first_part_events = first_part_events
        self.second_part_events = second_part_events

    def visualize_incomplete_process(self):
        output = ''
        for tmp_event in self.first_part_events:
            output += '(' + tmp_event + ')'
            output += '-->'
        output += '(???)-->'
        for tmp_event in self.second_part_events:
            output += '(' + tmp_event + ')'
            output += '-->'
        print(self.process_name.encode('utf-8'), ':', output[:-3].encode('utf-8'))


def Finding_analogous_processes_jaccard(target_process, candidate_process):
    first_part_length = len(target_process.first_part_events)
    second_part_length = len(target_process.second_part_events)
    first_part_score = calculate_jaccard_score(target_process.first_part_events,
                                               candidate_process.subevents[:first_part_length])
    second_part_score = calculate_jaccard_score(target_process.second_part_events,
                                                candidate_process.subevents[-second_part_length:])
    return first_part_score + second_part_score


def calculate_jaccard_score(events_1, events_2):
    event_1_tokens = list()
    event_2_tokens = list()
    for tmp_event in events_1:
        for w in tmp_event.split(' '):
            event_1_tokens.append(w)
    for tmp_event in events_2:
        for w in tmp_event.split(' '):
            event_2_tokens.append(w)
    event_1_tokens = set(event_1_tokens)
    event_2_tokens = set(event_2_tokens)
    joint_count = 0
    overall_count = len(event_1_tokens)
    for w in event_2_tokens:
        if w in event_1_tokens:
            joint_count += 1
        else:
            overall_count += 1
    if overall_count == 0:
        return 0
    else:
        return joint_count / overall_count


def filter_number_in_title(input_str):
    output_str = ''
    for c in input_str:
        if c not in '0123456789':
            output_str += c
    return output_str


def filter_string(input_str):
    output_str = ''
    for c in input_str:
        if c not in '\n':
            output_str += c
    return output_str


def raw_srl_to_structured_srl(raw_srl):
    # We only focus on the first verb (the main verb)
    # print(raw_srl)
    srl_string = raw_srl[0]
    components = list()
    new_component = list()
    for c in srl_string:
        if c == '[':
            new_component = list()
        elif c == ']':
            components.append(''.join(new_component))
        else:
            new_component.append(c)
    structured_srl = dict()
    for tmp_component in components:
        try:
            words = tmp_component.split(':')
            structured_srl[words[0]] = words[1][1:]
        except IndexError:
            continue
    return structured_srl


def shrinking_arguments(inital_structured_SRL):
    tmp_structure_for_task = dict()
    tmp_structure_for_task['V'] = ''
    tmp_structure_for_task['ARG0'] = ''
    tmp_structure_for_task['ARG1'] = ''
    if 'V' in inital_structured_SRL:
        tmp_structure_for_task['V'] = inital_structured_SRL['V']
    if 'ARG0' in inital_structured_SRL:
        tmp_s = spacy_model(inital_structured_SRL['ARG0'])
        for token in tmp_s:
            if token.dep_ == 'ROOT':
                tmp_structure_for_task['ARG0'] = token.lemma_
                break
    if 'ARG1' in inital_structured_SRL:
        tmp_s = spacy_model(inital_structured_SRL['ARG1'])
        for token in tmp_s:
            if token.dep_ == 'ROOT':
                tmp_structure_for_task['ARG1'] = token.lemma_
                break
    return tmp_structure_for_task


def sentence_to_event(SRL_parser, input_sentence):
    raw_result = SRL_parser.predict_json({'sentence': input_sentence})
    # print(raw_result)
    if len(raw_result['verbs']) > 0:
        # print(raw_srl_to_structured_srl(raw_result['verbs']))
        # print(shrinking_arguments(raw_srl_to_structured_srl(raw_result['verbs'])))
        return raw_result, shrinking_arguments(
            raw_srl_to_structured_srl([raw_result['verbs'][0]['description']]))
    else:
        return raw_result, {'V': '', 'ARG0': '', 'ARG1': ''}


def print_process(tmp_process):
    print('Title:', tmp_process['title_shrinked_srl'], 'ID:', tmp_process['id'])
    print('Subevnts:')
    for tmp_subevent in tmp_process['subevents']:
        print(tmp_subevent['shrinked_srl'])
    # print(tmp_process)


class SchemaNode:
    def __init__(self, initial_relations, process_id, event_id, pattern, verb_decay_rate, noun_decay_rate):
        self.parsed_relations = initial_relations.parsed_relations
        self.skeleton_parsed_relations = initial_relations.skeleton_parsed_relations
        self.skeleton_words = initial_relations.skeleton_words
        self.words = initial_relations.words
        self.process_id = process_id
        self.event_id = event_id
        self.pattern = pattern
        self.position_to_hypernym_path = self.get_position_to_hypernym_path(verb_decay_rate, noun_decay_rate)
        self.hypernym_combinations = list()
        self.cluster_name = pattern
        for tmp_p in self.position_to_hypernym_path:
            self.cluster_name += '$$'
            self.cluster_name += tmp_p
            self.cluster_name += ':'
            self.cluster_name += self.position_to_hypernym_path[tmp_p][0][0][0]

    def get_position_to_hypernym_path(self, verb_decay_rate, noun_decay_rate):
        tmp_dict = dict()
        left_words = self.skeleton_words
        position_to_graph_location = dict()
        while len(left_words) > 0:
            new_left_words = list()
            new_position_to_graph_location = dict()
            for w in left_words:
                tmp_count = 0
                found_upper_level = False
                for r in self.skeleton_parsed_relations:
                    if w[0] == r[2][0]:
                        tmp_count += 1
                        if r[0][0] in position_to_graph_location:
                            tmp_graph_location = position_to_graph_location[r[0][0]] + '$' + r[1]
                            new_position_to_graph_location[w[0]] = tmp_graph_location
                            tmp_dict[tmp_graph_location] = get_hypernym_path(w[1], verb_decay_rate, noun_decay_rate)
                            found_upper_level = True
                            break
                if tmp_count == 0:
                    new_position_to_graph_location[w[0]] = 'ROOT'
                    found_upper_level = True
                    tmp_dict['ROOT'] = get_hypernym_path(w[1], verb_decay_rate, noun_decay_rate)
                if not found_upper_level:
                    new_left_words.append(w)
            for new_pos in new_position_to_graph_location:
                position_to_graph_location[new_pos] = new_position_to_graph_location[new_pos]
            left_words = new_left_words
        return tmp_dict

    def get_predicate(self):
        for w in self.skeleton_words:
            if 'VB' in w[2]:
                return w[1]

    def to_string(self):
        all_words = list()
        for relation in self.parsed_relations:
            if relation[0] not in all_words:
                all_words.append(relation[0])
            if relation[2] not in all_words:
                all_words.append(relation[2])
        sorted_words = sorted(all_words, key=lambda tup: tup[0])
        generated_string = ''
        for tmp_word in sorted_words:
            generated_string += ' '
            generated_string += tmp_word[1]
        return generated_string[1:]

    def get_main_argument(self):
        for tmp_r in self.skeleton_parsed_relations:
            if tmp_r[1] == 'dobj':
                return tmp_r[2][1]
        for tmp_w in self.skeleton_words:
            if 'NN' in tmp_w[2]:
                return tmp_w[1]
        return 'NA'

    def update_hypernym_path(self, new_paths):
        self.hypernym_combinations = new_paths

    def update_cluster_name(self, new_cluster_name):
        self.cluster_name = new_cluster_name


def get_hypernym_path(input_word, verb_decay_rate=1.0, noun_decay_rate=1.0, max_length=4):
    paths = list()
    syn_sets = wn.synsets(input_word)
    for syn in syn_sets:
        raw_path = syn.hypernym_paths()
        for p in raw_path:
            tmp_path = [(input_word, 1)]
            last_node = (input_word, 1)
            for tmp_synset in p[::-1]:
                tmp_postag = tmp_synset._name.split('.')[1]
                if tmp_postag == 'v':
                    new_node = (tmp_synset._name, last_node[1] * verb_decay_rate)
                else:
                    new_node = (tmp_synset._name, last_node[1] * noun_decay_rate)
                tmp_path.append(new_node)
                last_node = new_node
            paths.append(tmp_path[:max_length])
    if len(paths) == 0:
        paths = [[(input_word, 1)]]
    return paths


def event_clustering_dp(nodes, position_keys):
    pattern_counting = dict()
    for tmp_node in nodes:
        raw_hypernym_list = list()
        for tmp_pos in position_keys:
            tmp_hypernyms = dict()
            for tmp_path in tmp_node.position_to_hypernym_path[tmp_pos]:
                for tmp_hypernym in tmp_path:
                    if tmp_hypernym[0] not in tmp_hypernyms:
                        tmp_hypernyms[tmp_hypernym[0]] = tmp_hypernym[1]
                    else:
                        if tmp_hypernyms[tmp_hypernym[0]] < tmp_hypernym[1]:
                            tmp_hypernyms[tmp_hypernym[0]] = tmp_hypernym[1]
            raw_hypernym_list.append(tmp_hypernyms)
        hypernym_combinations = [[]]
        for tmp_hypernym_list in raw_hypernym_list:
            tmp_hypernym_combinations = []
            for new_term in tmp_hypernym_list:
                for i in hypernym_combinations:
                    tmp_hypernym_combinations.append(i + [new_term])
            hypernym_combinations = tmp_hypernym_combinations
        hypernym_combinations_tuple = list()
        for tmp_combination in hypernym_combinations:
            tmp_combination_tuple = tuple(tmp_combination)
            if tmp_combination_tuple not in pattern_counting:
                pattern_counting[tmp_combination_tuple] = 0
            overall_score = 1
            for i, tmp_term in enumerate(tmp_combination_tuple):
                overall_score *= raw_hypernym_list[i][tmp_term]
            pattern_counting[tmp_combination_tuple] += overall_score
            hypernym_combinations_tuple.append(tmp_combination_tuple)
        tmp_node.update_hypernym_path(hypernym_combinations_tuple)
    sorted_cluster_names = sorted(pattern_counting, key=lambda x: pattern_counting[x], reverse=True)
    current_largest_cluster = sorted_cluster_names[0]
    covered_nodes = list()
    uncovered_nodes = list()
    cluster_weight = dict()
    key_and_words = list()
    for i, tmp_key in enumerate(position_keys):
        key_and_words.append(tmp_key + ':' + current_largest_cluster[i])
    cluster_weight['$$'.join(key_and_words)] = pattern_counting[current_largest_cluster]
    for tmp_node in nodes:
        if current_largest_cluster in tmp_node.hypernym_combinations:
            tmp_node.update_cluster_name('$$'.join(key_and_words))
            covered_nodes.append(tmp_node)
        else:
            uncovered_nodes.append(tmp_node)
    if len(uncovered_nodes) > 0:
        uncovered_nodes, other_cluster_weights = event_clustering_dp(uncovered_nodes, position_keys)
        for tmp_c in other_cluster_weights:
            cluster_weight[tmp_c] = other_cluster_weights[tmp_c]
    return covered_nodes + uncovered_nodes, cluster_weight


def event_clustering(input_event_sequences, w_v, w_a):
    # print('We are clustering all events')
    clusters_based_on_pattern = dict()
    nodes = list()
    nodes_by_sequences = list()
    for i, tmp_event_sequence in enumerate(input_event_sequences):
        tmp_sequence = list()
        for j, tmp_event in enumerate(tmp_event_sequence):
            tmp_node = SchemaNode(tmp_event, i, j, tmp_event.pattern, w_v, w_a)
            nodes.append(tmp_node)
            tmp_sequence.append(tmp_node)
        nodes_by_sequences.append(tmp_sequence)

    for tmp_node in nodes:
        if tmp_node.pattern not in clusters_based_on_pattern:
            clusters_based_on_pattern[tmp_node.pattern] = list()
        clusters_based_on_pattern[tmp_node.pattern].append(tmp_node)
    new_nodes = list()
    cluster_weight = dict()
    for tmp_pattern in clusters_based_on_pattern:
        tmp_nodes = clusters_based_on_pattern[tmp_pattern]
        tmp_position_keys = list()
        for tmp_key in tmp_nodes[0].position_to_hypernym_path:
            tmp_position_keys.append(tmp_key)
        tmp_position_keys = sorted(tmp_position_keys, key=lambda x: len(x))
        nodes_after_clustering, tmp_cluster_weight = event_clustering_dp(tmp_nodes, tuple(tmp_position_keys))
        for tmp_node in nodes_after_clustering:
            tmp_node.update_cluster_name(tmp_pattern + '$$' + tmp_node.cluster_name)
            new_nodes.append(tmp_node)
        for tmp_c in tmp_cluster_weight:
            cluster_weight[tmp_pattern + '$$' + tmp_c] = tmp_cluster_weight[tmp_c]

    new_sequence = list()
    for i in range(len(input_event_sequences)):
        new_sequence.append(list())

    clusters = dict()
    for tmp_node in nodes:
        if tmp_node.cluster_name not in clusters:
            clusters[tmp_node.cluster_name] = list()
        clusters[tmp_node.cluster_name].append(tmp_node)
        new_sequence[tmp_node.process_id].append(tmp_node)
    finalized_sequence = list()
    for tmp_p in new_sequence:
        new_p = sorted(tmp_p, key=lambda x: x.event_id)
        finalized_sequence.append(new_p)
    return new_nodes, clusters, finalized_sequence, cluster_weight


def schema_induction(input_event_sequences, w_v, w_a):
    # load activities into schema nodes
    nodes, clusters, nodes_by_sequences, cluster_weight = event_clustering(input_event_sequences, w_v, w_a)

    # sort all clusters
    pairwise_front_counting = dict()
    for tmp_c_1 in clusters:
        for tmp_c_2 in clusters:
            pairwise_front_counting[tmp_c_1 + '&&' + tmp_c_2] = 0
    for tmp_event_sequence in nodes_by_sequences:
        for i, tmp_node1 in enumerate(tmp_event_sequence):
            for j, tmp_node2 in enumerate(tmp_event_sequence):
                if i < j:
                    pairwise_front_counting[tmp_node1.cluster_name + '&&' + tmp_node2.cluster_name] += 1
    front_counting = dict()
    for tmp_c in clusters:
        front_counting[tmp_c] = 0
    for tmp_c_1 in clusters:
        for tmp_c_2 in clusters:
            if tmp_c_1 != tmp_c_2:
                if pairwise_front_counting[tmp_c_1 + '&&' + tmp_c_2] > \
                        pairwise_front_counting[tmp_c_2 + '&&' + tmp_c_1]:
                    front_counting[tmp_c_1] += 1
                if pairwise_front_counting[tmp_c_1 + '&&' + tmp_c_2] < \
                        pairwise_front_counting[tmp_c_2 + '&&' + tmp_c_1]:
                    front_counting[tmp_c_2] += 1
                if pairwise_front_counting[tmp_c_1 + '&&' + tmp_c_2] == \
                        pairwise_front_counting[tmp_c_2 + '&&' + tmp_c_1]:
                    front_counting[tmp_c_1] += 0
                    front_counting[tmp_c_2] += 0
    sorted_group = sorted(front_counting, key=lambda x: front_counting[x], reverse=True)
    process_structure = list()
    for i, tmp_c in enumerate(sorted_group):
        process_structure.append(
            {'name': tmp_c, 'location': (i + 1) / len(sorted_group), 'weight': cluster_weight[tmp_c],
             'raw_location': front_counting[tmp_c]})
    return process_structure
    # for tmp_c in sorted_group:
    #     if len(clusters[tmp_c]) < 3:
    #         continue
    #     print('---'*32)
    #     print('cluster name:', tmp_c)
    #     print('Position score:', front_counting[tmp_c])
    #     print('contained events:')
    #     for tmp_event in clusters[tmp_c]:
    #         print(tmp_event.to_string())
    # print('number of groups:', len(sorted_group))


def event_structure_to_string(input_structure):
    tmp_words = list()
    for tmp_position in input_structure.split('$$')[1:]:
        tmp_words.append(tmp_position.split(':')[1].split('.')[0])
    return ' '.join(tmp_words)


def compute_event_semantics_similarity(reference_event, prediction_event):
    reference_event_type = reference_event.split('$$')[0]
    prediction_event_type = prediction_event.split('$$')[0]
    if reference_event_type != prediction_event_type:
        return 0
    structured_reference_event = dict()
    for tmp_w in reference_event.split('$$')[1:]:
        # if tmp_w.split(':')[0] == 'ROOT':
        #     structured_reference_event[tmp_w.split(':')[0]] = tmp_w.split(':')[1]
        structured_reference_event[tmp_w.split(':')[0]] = tmp_w.split(':')[1]
    structured_prediction_event = dict()
    for tmp_w in prediction_event.split('$$')[1:]:
        # if tmp_w.split(':')[0] == 'ROOT':
        #     structured_prediction_event[tmp_w.split(':')[0]] = tmp_w.split(':')[1]
        structured_prediction_event[tmp_w.split(':')[0]] = tmp_w.split(':')[1]
    tmp_score = 1
    for tmp_position in structured_reference_event:
        reference_w = structured_reference_event[tmp_position].split('.')[0]
        prediction_w = structured_prediction_event[tmp_position].split('.')[0]

        tmp_hypernym_path = get_hypernym_path(reference_w, 0, 0)
        matching_scores = list()
        for tmp_p in tmp_hypernym_path:
            matching_score = 0
            for tmp_t in tmp_p:
                if prediction_w.split('.')[0] == tmp_t[0].split('.')[0]:
                    matching_score = tmp_t[1]
                    break
            matching_scores.append(matching_score)
        prediction_score = max(matching_scores)

        tmp_hypernym_path = get_hypernym_path(prediction_w, 0, 0)
        matching_scores = list()
        for tmp_p in tmp_hypernym_path:
            matching_score = 0
            for tmp_t in tmp_p:
                if reference_w.split('.')[0] == tmp_t[0].split('.')[0]:
                    matching_score = tmp_t[1]
                    break
            matching_scores.append(matching_score)
        reference_score = max(matching_scores)
        tmp_score *= max((prediction_score, reference_score))
    return tmp_score


def compute_sequence_semantics_similarity(reference_sequence, prediction_sequence):
    overall_score = 1
    for i in range(len(reference_sequence)):
        reference_event = reference_sequence[i]
        prediction_event = prediction_sequence[i]
        overall_score *= compute_event_semantics_similarity(reference_event, prediction_event)
    return overall_score


def get_matching_score(reference_set, prediction_set):
    total_scores = 0
    for tmp_prediction in prediction_set:
        tmp_scores = list()
        for tmp_reference in reference_set:
            tmp_scores.append(compute_sequence_semantics_similarity(tmp_reference, tmp_prediction))
        try:
            total_scores += max(tmp_scores)
        except:
            print(reference_set)
            print(prediction_set)
            exit(1)
    return total_scores


def compute_event_bleu(reference_structure, predicted_structure, max_order=3):

    matching_scores = list()
    reference_lengths = list()
    prediction_lengths = list()
    orders = [1, 2]
    for tmp_order in orders:
        reference_set = list()
        prediction_set = list()
        if tmp_order == 1:
            for i in range(0, len(reference_structure) - tmp_order + 1):
                reference_set.append(reference_structure[i:i + tmp_order])
            for i in range(0, len(predicted_structure) - tmp_order + 1):
                prediction_set.append(predicted_structure[i:i + tmp_order])
        elif tmp_order == 2:
            for i in range(0, len(reference_structure) - tmp_order + 1):
                for j in range(1, len(reference_structure)):
                    reference_set.append([reference_structure[i], reference_structure[j]])
            for i in range(0, len(predicted_structure) - tmp_order + 1):
                for j in range(1, len(predicted_structure)):
                    prediction_set.append([predicted_structure[i], predicted_structure[j]])
        else:
            for i in range(0, len(reference_structure) - tmp_order + 1):
                reference_set.append(reference_structure[i:i + tmp_order])
            for i in range(0, len(predicted_structure) - tmp_order + 1):
                prediction_set.append(predicted_structure[i:i + tmp_order])

        reference_lengths.append(len(reference_set))
        prediction_lengths.append(len(prediction_set))
        if len(reference_set) > 0 and len(prediction_set) > 0:
            matching_score = get_matching_score(reference_set, prediction_set)
        else:
            matching_score = 0
        matching_scores.append(matching_score)

    return matching_scores, reference_lengths, prediction_lengths


def compute_event_bleu_multi_reference(reference_structures, predicted_structure, max_order=3):
    matching_scores = list()
    reference_lengths = list()
    prediction_lengths = list()
    orders = [1, 2]
    for tmp_order in orders:
        reference_set = list()
        prediction_set = list()
        if tmp_order == 1:
            for tmp_reference in reference_structures:
                for i in range(0, len(tmp_reference) - tmp_order + 1):
                    reference_set.append(tmp_reference[i:i + tmp_order])
            for i in range(0, len(predicted_structure) - tmp_order + 1):
                prediction_set.append(predicted_structure[i:i + tmp_order])
        elif tmp_order == 2:
            for tmp_reference in reference_structures:
                for i in range(0, len(tmp_reference) - tmp_order + 1):
                    for j in range(1, len(tmp_reference)):
                        reference_set.append([tmp_reference[i], tmp_reference[j]])

            for i in range(0, len(predicted_structure) - tmp_order + 1):
                for j in range(1, len(predicted_structure)):
                    prediction_set.append([predicted_structure[i], predicted_structure[j]])
        else:
            for tmp_reference in reference_structures:
                for i in range(0, len(tmp_reference) - tmp_order + 1):
                    reference_set.append(tmp_reference[i:i + tmp_order])
            for i in range(0, len(predicted_structure) - tmp_order + 1):
                prediction_set.append(predicted_structure[i:i + tmp_order])

        reference_lengths.append(len(reference_set))
        prediction_length = 0
        for tmp_set in prediction_set:
            if 'EOS' not in tmp_set:
                prediction_length += 1
        prediction_lengths.append(prediction_length)
        if len(reference_set) > 0 and len(prediction_set) > 0:
            matching_score = get_matching_score(reference_set, prediction_set)
        else:
            matching_score = 0
        matching_scores.append(matching_score)

    return matching_scores, reference_lengths, prediction_lengths


def evaluate_multi_reference(test_data, prediction):
    test_data_by_key = dict()
    for tmp_process in test_data:
        if tmp_process['process_key'] not in test_data_by_key:
            test_data_by_key[tmp_process['process_key']] = list()
        test_data_by_key[tmp_process['process_key']].append(tmp_process['subevent_structures'])
    degree_1_match = 0
    degree_1_reference = 0
    degree_1_prediction = 0
    degree_2_match = 0
    degree_2_reference = 0
    degree_2_prediction = 0
    for tmp_process_key in tqdm(test_data_by_key):
        tmp_references = test_data_by_key[tmp_process_key]
        # tmp_prediction = prediction[tmp_process['title']]
        tmp_prediction = prediction[tmp_process_key]
        tmp_matching_scores, tmp_reference_lengths, tmp_prediction_lengths = compute_event_bleu_multi_reference(tmp_references, tmp_prediction)
        degree_1_match += tmp_matching_scores[0]
        degree_1_reference += tmp_reference_lengths[0]
        degree_1_prediction += tmp_prediction_lengths[0]

        degree_2_match += tmp_matching_scores[1]
        degree_2_reference += tmp_reference_lengths[1]
        degree_2_prediction += tmp_prediction_lengths[1]
    print('EBLEU 1:', degree_1_match / degree_1_prediction)
    # print('recall 1:', degree_1_match / degree_1_reference)
    print('EBLEU 2:', degree_2_match / degree_2_prediction)
    # print('recall 2:', degree_2_match / degree_2_reference)
    return {'EBLEU1': degree_1_match / degree_1_prediction, 'EBLEU2': degree_2_match / degree_2_prediction}


def select_example(test_data, prediction):
    test_data_by_key = dict()
    for tmp_process in test_data:
        if tmp_process['process_key'] not in test_data_by_key:
            test_data_by_key[tmp_process['process_key']] = list()
        test_data_by_key[tmp_process['process_key']].append(tmp_process['subevent_structures'])
    degree_1_match = 0
    degree_1_reference = 0
    degree_1_prediction = 0
    degree_2_match = 0
    degree_2_reference = 0
    degree_2_prediction = 0
    key_to_ebleu1 = dict()
    key_to_ebleu2 = dict()
    for tmp_process_key in tqdm(test_data_by_key):
        tmp_references = test_data_by_key[tmp_process_key]
        # tmp_prediction = prediction[tmp_process['title']]
        tmp_prediction = prediction[tmp_process_key]
        tmp_matching_scores, tmp_reference_lengths, tmp_prediction_lengths = compute_event_bleu_multi_reference(tmp_references, tmp_prediction)
        degree_1_match += tmp_matching_scores[0]
        degree_1_reference += tmp_reference_lengths[0]
        degree_1_prediction += tmp_prediction_lengths[0]
        if tmp_prediction_lengths[0] > 0:
            key_to_ebleu1[tmp_process_key] = tmp_matching_scores[0]/tmp_prediction_lengths[0]
        else:
            key_to_ebleu1[tmp_process_key] = 0
        degree_2_match += tmp_matching_scores[1]
        degree_2_reference += tmp_reference_lengths[1]
        degree_2_prediction += tmp_prediction_lengths[1]
        if tmp_prediction_lengths[1] > 0:
            key_to_ebleu2[tmp_process_key] = tmp_matching_scores[1]/tmp_prediction_lengths[1]
        else:
            key_to_ebleu2[tmp_process_key] = 0
    # print('precision 1:', degree_1_match / degree_1_prediction)
    # print('recall 1:', degree_1_match / degree_1_reference)
    # print('precision 2:', degree_2_match / degree_2_prediction)
    # print('recall 2:', degree_2_match / degree_2_reference)
    return key_to_ebleu1, key_to_ebleu2, test_data_by_key

def evaluate_precition_and_recall(test_data, prediction):
    degree_1_match = 0
    degree_1_reference = 0
    degree_1_prediction = 0
    degree_2_match = 0
    degree_2_reference = 0
    degree_2_prediction = 0
    for tmp_process in tqdm(test_data):
        tmp_reference = tmp_process['subevent_structures']
        # tmp_prediction = prediction[tmp_process['title']]
        tmp_prediction = prediction[tmp_process['process_key']]
        tmp_matching_scores, tmp_reference_lengths, tmp_prediction_lengths = compute_event_bleu(tmp_reference, tmp_prediction)
        degree_1_match += tmp_matching_scores[0]
        degree_1_reference += tmp_reference_lengths[0]
        degree_1_prediction += tmp_prediction_lengths[0]

        degree_2_match += tmp_matching_scores[1]
        degree_2_reference += tmp_reference_lengths[1]
        degree_2_prediction += tmp_prediction_lengths[1]
    print('precision 1:', degree_1_match / degree_1_prediction)
    print('recall 1:', degree_1_match / degree_1_reference)
    print('precision 2:', degree_2_match / degree_2_prediction)
    print('recall 2:', degree_2_match / degree_2_reference)


def evaluate_human_performance_multi_reference(test_data):
    test_data_by_key = dict()
    for tmp_process in test_data:
        if tmp_process['process_key'] not in test_data_by_key:
            test_data_by_key[tmp_process['process_key']] = list()
        test_data_by_key[tmp_process['process_key']].append(tmp_process['subevent_structures'])
    degree_1_match = 0
    degree_1_reference = 0
    degree_1_prediction = 0
    degree_2_match = 0
    degree_2_reference = 0
    degree_2_prediction = 0
    for tmp_process_key in tqdm(test_data_by_key):
        all_references = test_data_by_key[tmp_process_key]
        if len(all_references) < 2:
            continue
        if len(all_references) > 10:
            continue
        for i in range(len(all_references)):
            tmp_prediction = all_references[i]
            tmp_references = list()
            for j in range(len(all_references)):
                if i != j:
                    tmp_references.append(all_references[j])
            tmp_matching_scores, tmp_reference_lengths, tmp_prediction_lengths = compute_event_bleu_multi_reference(tmp_references, tmp_prediction)
            degree_1_match += tmp_matching_scores[0]
            degree_1_reference += tmp_reference_lengths[0]
            degree_1_prediction += tmp_prediction_lengths[0]

            degree_2_match += tmp_matching_scores[1]
            degree_2_reference += tmp_reference_lengths[1]
            degree_2_prediction += tmp_prediction_lengths[1]
    print('precision 1:', degree_1_match / degree_1_prediction)
    print('recall 1:', degree_1_match / degree_1_reference)
    print('precision 2:', degree_2_match / degree_2_prediction)
    print('recall 2:', degree_2_match / degree_2_reference)

def merge_event_for_prediction(word_to_hypernym_paths, main_event, detailed_event, normalize_weight, w_v, w_n):
    detailed_event_words_paths = list()

    for tmp_w in detailed_event['name'].split('$$')[1:]:
        # tmp_argument_w = tmp_w.split(':')[1].split('.')[0]
        # tmp_paths = get_hypernym_path(tmp_argument_w, w_v, w_n)
        # for tmp_path in tmp_paths:
        #     detailed_event_words_paths.append(tmp_path)
        tmp_argument_w = tmp_w.split(':')[1].split('.')[0]
        # if tmp_argument_w not in word_to_hypernym_paths:
        #     tmp_paths = get_hypernym_path(tmp_argument_w, w_v, w_n)
        #     word_to_hypernym_paths[tmp_argument_w] = tmp_paths
        # else:
        #     tmp_paths = word_to_hypernym_paths[tmp_argument_w]
        tmp_paths = word_to_hypernym_paths[tmp_argument_w]
        for tmp_path in tmp_paths:
            detailed_event_words_paths.append(tmp_path)

    main_event_type = main_event['name'].split('$$')[0]
    position2synset = dict()
    for tmp_w in main_event['name'].split('$$')[1:]:
        tmp_position = tmp_w.split(':')[0]
        tmp_predicate_w = tmp_w.split(':')[1].split('.')[0]
        instantiation_dict = dict()
        instantiation_dict[tmp_predicate_w] = 1
        for detailed_path in detailed_event_words_paths:
            found_match = False
            for w in detailed_path:
                if tmp_predicate_w == w[0].split('.')[0]:
                    if detailed_path[0][0] not in instantiation_dict:
                        instantiation_dict[detailed_path[0][0]] = 0
                    instantiation_dict[detailed_path[0][0]] += w[1]
                    found_match = True
                    break
            if found_match:
                break
        sorted_new_words = sorted(instantiation_dict, key=lambda x: instantiation_dict[x], reverse=True)
        position2synset[tmp_position] = sorted_new_words[0]
    result_event = ''
    result_event += main_event_type
    for tmp_position in position2synset:
        result_event += '$$'
        result_event += tmp_position
        result_event += ':'
        result_event += position2synset[tmp_position]
    new_event = dict()
    new_event['name'] = result_event
    new_event['weight'] = main_event['weight'] * detailed_event['weight'] / normalize_weight
    new_event['location'] = main_event['location']
    new_event['raw_location'] = main_event['raw_location']
    return new_event


def predict_structure(word_to_hypernym_paths, predicate_subevents, argument_subevents, number_of_subevents, w_v, w_n, prediction_method='main'):
    if prediction_method == 'basic':
        all_nodes = predicate_subevents + argument_subevents
        sorted_structure = sorted(all_nodes, key=lambda x: x['weight'], reverse=True)
        selected_nodes = sorted_structure[:number_of_subevents]
        sorted_structure = sorted(selected_nodes, key=lambda x: x['location'])
        final_structure = list()
        for tmp_node in sorted_structure:
            final_structure.append(tmp_node['name'])
        return final_structure
    elif prediction_method == 'normalize':
        predicate_weight_sum = 0
        for tmp_e in predicate_subevents:
            predicate_weight_sum += tmp_e['weight']
        new_predicate_subevents = list()
        for tmp_e in predicate_subevents:
            tmp_e['weight'] /= predicate_weight_sum
            # tmp_e['weight'] /= number_of_arguments
            new_predicate_subevents.append(tmp_e)
        argument_weight_sum = 0
        for tmp_e in argument_subevents:
            argument_weight_sum += tmp_e['weight']
        new_argument_subevents = list()
        for tmp_e in argument_subevents:
            tmp_e['weight'] /= argument_weight_sum
            # tmp_e['weight'] /= number_of_predicates
            new_argument_subevents.append(tmp_e)
        all_nodes = predicate_subevents + argument_subevents
        sorted_structure = sorted(all_nodes, key=lambda x: x['weight'], reverse=True)
        selected_nodes = sorted_structure[:number_of_subevents]
        sorted_structure = sorted(selected_nodes, key=lambda x: x['location'])
        final_structure = list()
        for tmp_node in sorted_structure:
            final_structure.append(tmp_node['name'])
        return final_structure
    elif prediction_method == 'half-half':
        number_of_argument_events = int(number_of_subevents * 0.5)
        number_of_predicate_events = number_of_subevents - number_of_argument_events
        sorted_predicate_events = sorted(predicate_subevents, key=lambda x: x['weight'], reverse=True)
        sorted_argument_events = sorted(argument_subevents, key=lambda x: x['weight'], reverse=True)
        all_nodes = sorted_predicate_events[:number_of_predicate_events] + sorted_argument_events[
                                                                           :number_of_argument_events]
        sorted_structure = sorted(all_nodes, key=lambda x: x['location'])
        final_structure = list()
        for tmp_node in sorted_structure:
            final_structure.append(tmp_node['name'])
        return final_structure
    elif prediction_method == 'main':
        # We are going to implement the main merging algorithm here.
        all_new_events = list()
        predicate_weight_sum = 0
        for tmp_e in predicate_subevents:
            predicate_weight_sum += tmp_e['weight']
        argument_weight_sum = 0
        for tmp_e in argument_subevents:
            argument_weight_sum += tmp_e['weight']
        for tmp_predicate_event in predicate_subevents:
            for tmp_argument_event in argument_subevents:
                new_predicate_event = merge_event_for_prediction(word_to_hypernym_paths, tmp_predicate_event, tmp_argument_event, argument_weight_sum, w_v, w_n)
                new_argument_event = merge_event_for_prediction(word_to_hypernym_paths, tmp_argument_event, tmp_predicate_event, predicate_weight_sum, w_v, w_n)
                all_new_events.append(new_predicate_event)
                all_new_events.append(new_argument_event)
        # we need to merge events
        new_events_by_name = dict()
        for tmp_e in all_new_events:
            if tmp_e['name'] not in new_events_by_name:
                new_events_by_name[tmp_e['name']] = list()
            new_events_by_name[tmp_e['name']].append(tmp_e)
        merged_nodes = list()
        for tmp_name in new_events_by_name:
            overall_location = 0
            overall_weight = 0
            for tmp_e in new_events_by_name[tmp_name]:
                overall_location += tmp_e['location'] * tmp_e['weight']
                overall_weight += tmp_e['weight']
            overall_location = overall_location/overall_weight
            merged_nodes.append({'name': tmp_name, 'location': overall_location, 'weight': overall_weight})
        sorted_structure = sorted(merged_nodes, key=lambda x: x['weight'], reverse=True)
        selected_nodes = sorted_structure[:number_of_subevents]
        # selected_nodes = nodes_after_filtering
        sorted_structure = sorted(selected_nodes, key=lambda x: x['location'])
        # random.shuffle(selected_nodes)
        # sorted_structure = selected_nodes
        final_structure = list()
        for tmp_node in sorted_structure:
            final_structure.append(tmp_node['name'])
        return final_structure


spacy_model = spacy.load("en_core_web_sm")
wn = WordNet()




spacy_model = spacy.load("en_core_web_sm")
