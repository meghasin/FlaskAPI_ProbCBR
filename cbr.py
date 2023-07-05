import argparse
import numpy as np
from scipy.special import logsumexp
import os
from tqdm import tqdm
import scipy.sparse
from collections import defaultdict, deque
import pickle
import torch
import uuid
from typing import *
import logging
import json
import pandas as pd
import sys
#import wandb
from scipy.stats import rankdata
from ext.preprocessing import combine_path_splits
from ext.utils import get_programs, create_sparse_adj_mats, execute_one_program
from ext.data.data_utils import create_vocab, load_vocab, load_data, get_unique_entities, \
    read_graph, get_entities_group_by_relation, get_inv_relation, load_data_all_triples, create_adj_list

# logger = logging.getLogger()
# logging.basicConfig(
#     format="%(asctime)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO
# )

MRN_nodes = pd.read_csv('/home/msinha/CBR-AKBC/nodes_biolink.csv', dtype=str)
class ProbCBR(object):
    def __init__(self, args, train_map, full_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab,
                 eval_rev_vocab, all_paths, rel_ent_map, per_relation_config: Union[None, dict],ans_num):
        self.args = args
        self.eval_map = eval_map
        self.train_map = train_map
        self.full_map = full_map
        self.all_zero_ctr = []
        self.ans_num = ans_num
        self.all_num_ret_nn = []
        self.entity_vocab, self.rev_entity_vocab, self.rel_vocab, self.rev_rel_vocab = entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab
        self.eval_vocab, self.eval_rev_vocab = eval_vocab, eval_rev_vocab
        self.all_paths = all_paths
        self.rel_ent_map = rel_ent_map
        self.per_relation_config = per_relation_config
        self.num_non_executable_programs = []
        self.nearest_neighbor_1_hop = None
        #logger.info("Building sparse adjacency matrices")
        self.sparse_adj_mats = create_sparse_adj_mats(self.train_map, self.entity_vocab, self.rel_vocab)
        self.top_query_preds = {}

    def set_nearest_neighbor_1_hop(self, nearest_neighbor_1_hop):
        self.nearest_neighbor_1_hop = nearest_neighbor_1_hop

    def get_nearest_neighbor_inner_product(self, e1: str, r: str, k: Optional[int] = 5) -> Union[List[str], None]:
        try:
            nearest_entities = [self.rev_entity_vocab[e] for e in
                                self.nearest_neighbor_1_hop[self.eval_vocab[e1]].tolist()]
            # remove e1 from the set of k-nearest neighbors if it is there.
            nearest_entities = [nn for nn in nearest_entities if nn != e1]
            # making sure, that the similar entities also have the query relation
            ctr = 0
            temp = []
            for nn in nearest_entities:
                if ctr == k:
                    break
                if len(self.train_map[nn, r]) > 0:
                    temp.append(nn)
                    ctr += 1
            nearest_entities = temp
        except KeyError:
            return None
        return nearest_entities

    def get_programs_from_nearest_neighbors(self, e1: str, r: str, nn_func: Callable, num_nn: Optional[int] = 5):
        all_programs = []
        nearest_entities = nn_func(e1, r, k=num_nn)
        use_cheat_neighbors_for_r = self.args.cheat_neighbors if self.per_relation_config is None else \
            self.per_relation_config[r]["cheat_neighbors"]
        if (nearest_entities is None or len(nearest_entities) == 0) and use_cheat_neighbors_for_r:
            num_ent_with_r = len(self.rel_ent_map[r])
            if num_ent_with_r > 0:
                if num_ent_with_r < num_nn:
                    nearest_entities = self.rel_ent_map[r]
                else:
                    random_idx = np.random.choice(num_ent_with_r, num_nn, replace=False)
                    nearest_entities = [self.rel_ent_map[r][r_idx] for r_idx in random_idx]
        if nearest_entities is None or len(nearest_entities) == 0:
            self.all_num_ret_nn.append(0)
            return []
        self.all_num_ret_nn.append(len(nearest_entities))
        zero_ctr = 0
        for e in nearest_entities:
            if len(self.train_map[(e, r)]) > 0:
                paths_e = self.all_paths[e]  # get the collected 3 hop paths around e
                nn_answers = self.train_map[(e, r)]
                for nn_ans in nn_answers:
                    all_programs += get_programs(e, nn_ans, paths_e)
            elif len(self.train_map[(e, r)]) == 0:
                zero_ctr += 1
        self.all_zero_ctr.append(zero_ctr)
        return all_programs

    def rank_programs(self, list_programs: List[List[str]], r: str) -> List[List[str]]:
        """
        Rank programs.
        """
        # sort it by the path score
        unique_programs = set()
        for p in list_programs:
            unique_programs.add(tuple(p))
        # now get the score of each path
        path_and_scores = []
        use_only_precision_scores_for_r = self.args.use_only_precision_scores if self.per_relation_config is None \
            else self.per_relation_config[r]["use_only_precision_scores"]
        for p in unique_programs:
            try:
                if use_only_precision_scores_for_r:
                    path_and_scores.append((p, self.args.precision_map[self.c][r][p]))
                else:
                    path_and_scores.append((p, self.args.path_prior_map_per_relation[self.c][r][p] *
                                            self.args.precision_map[self.c][r][p]))
            except KeyError:
                # TODO: Fix key error
                if len(p) == 1 and p[0] == r:
                    continue  # ignore query relation
                else:
                    # use the fall back score
                    try:
                        c = 0
                        if use_only_precision_scores_for_r:
                            score = self.args.precision_map_fallback[c][r][p]
                        else:
                            score = self.args.path_prior_map_per_relation_fallback[c][r][p] * \
                                    self.args.precision_map_fallback[c][r][p]
                        path_and_scores.append((p, score))
                    except KeyError:
                        # still a path or rel is missing.
                        path_and_scores.append((p, 0))

        # sort wrt counts
        json1 = json.dumps(path_and_scores)
        #f = open("path_and_scores.json","w")
        #f.write(json1)
        #f.close()
        
        sorted_programs = [k for k, v in sorted(path_and_scores, key=lambda item: -item[1]) if float(v) != 0.0]
        sorted_programs_with_score = [(k,float(v)) for k, v in sorted(path_and_scores, key=lambda item: -item[1]) if float(v) !=         0.0]

        return sorted_programs, sorted_programs_with_score

    def execute_programs(self, e: str, r: str, path_list: List[List[str]], max_branch: Optional[int] = 1000) \
            -> Tuple[List[Tuple[np.ndarray, float, List[str]]], List[List[str]]]:

        def _fall_back(r, p):
            """
            When a cluster does not have a query relation (because it was not seen during counting)
            or if a path is not found, then fall back to no cluster statistics
            :param r:
            :param p:
            :return:
            """
            c = 0  # one cluster for all entity
            try:
                score = self.args.path_prior_map_per_relation_fallback[c][r][p] * \
                        self.args.precision_map_fallback[c][r][p]
            except KeyError:
                # either the path or relation is missing from the fall back map as well
                score = 0
            return score

        all_answers = []
        not_executed_paths = []
        execution_fail_counter = 0
        executed_path_counter = 0
        max_num_programs_for_r = self.args.max_num_programs if self.per_relation_config is None else \
            self.per_relation_config[r]["max_num_programs"]
        for path in path_list:
            if executed_path_counter == max_num_programs_for_r:
                break
            ans = execute_one_program(self.sparse_adj_mats, self.entity_vocab, e, path)
            if self.args.use_path_counts:
                try:
                    if path in self.args.path_prior_map_per_relation[self.c][r] and path in \
                            self.args.precision_map[self.c][r]:
                        path_score = self.args.path_prior_map_per_relation[self.c][r][path] * \
                                     self.args.precision_map[self.c][r][path]
                    else:
                        # logger.info("This path was not there in the cluster for the relation.")
                        path_score = _fall_back(r, path)
                except KeyError:
                    # logger.info("Looks like the relation was not found in the cluster, have to fall back")
                    # fallback to the global scores
                    path_score = _fall_back(r, path)
            else:
                path_score = 1
            path = tuple(path)
            if len(np.nonzero(ans)[0]) == 0:
                not_executed_paths.append(path)
                execution_fail_counter += 1
            else:
                executed_path_counter += 1
            all_answers += [(ans, path_score, path)]
        #np.savetxt("all_answers.csv", all_answers, delimiter=",", fmt='%s')
        self.num_non_executable_programs.append(execution_fail_counter)
        return all_answers, not_executed_paths

    def rank_answers(self, list_answers: List[Tuple[np.ndarray, float, List[str]]], aggr_type1="none",
                     aggr_type2="sum") -> List[
        str]:
        """
        Different ways to re-rank answers
        """

        def rank_entities_by_max_score(score_map):
            """
            sorts wrt top value. If there are same values, then it sorts wrt the second value
            :param score_map:
            :return:
            """
            # sort wrt the max value
            if len(score_map) == 0:
                return []
            sorted_score_map = sorted(score_map.items(), key=lambda kv: -kv[1][0])
            sorted_score_map_second_round = []
            temp = []
            curr_val = sorted_score_map[0][1][0]  # value of the first
            for (k, v) in sorted_score_map:
                if v[0] == curr_val:
                    temp.append((k, v))
                else:
                    sorted_temp = sorted(temp, key=lambda kv: -kv[1][1] if len(
                        kv[1]) > 1 else 1)  # sort wrt second highest score
                    sorted_score_map_second_round += sorted_temp
                    temp = [(k, v)]  # clear temp and add new val
                    curr_val = v[0]  # calculate new curr_val
            # do the same for remaining elements in temp
            if len(temp) > 0:
                sorted_temp = sorted(temp,
                                     key=lambda kv: -kv[1][1] if len(kv[1]) > 1 else 1)  # sort wrt second highest score
                sorted_score_map_second_round += sorted_temp
            return sorted_score_map_second_round

        count_map = {}
        uniq_entities = set()
        for e_vec, e_score, path in list_answers:
            path_answers = [(self.rev_entity_vocab[d_e], e_vec[d_e]) for d_e in np.nonzero(e_vec)[0]]
            for e, e_c in path_answers:
                if e not in count_map:
                    count_map[e] = {}
                if aggr_type1 == "none":
                    count_map[e][path] = e_score  # just count once for a path type.
                elif aggr_type1 == "sum":
                    count_map[e][path] = e_score * e_c  # aggregate for each path
                else:
                    raise NotImplementedError("{} aggr_type1 is invalid".format(aggr_type1))
                uniq_entities.add(e)
        score_map = defaultdict(int)
        for e, path_scores_map in count_map.items():
            p_scores = [v for k, v in path_scores_map.items()]
            if aggr_type2 == "sum":
                score_map[e] = np.sum(p_scores)
            elif aggr_type2 == "max":
                score_map[e] = sorted(p_scores, reverse=True)
            elif aggr_type2 == "noisy_or":
                score_map[e] = 1 - np.prod(1 - np.asarray(p_scores))
            elif aggr_type2 == "logsumexp":
                score_map[e] = logsumexp(p_scores)
            else:
                raise NotImplementedError("{} aggr_type2 is invalid".format(aggr_type2))
        if aggr_type2 == "max":
            sorted_entities_by_val = rank_entities_by_max_score(score_map)
        else:
            sorted_entities_by_val = sorted(score_map.items(), key=lambda kv: -kv[1])
        return sorted_entities_by_val

    @staticmethod
    def get_rank_in_list(e, predicted_answers):
        predicted_answers = pd.DataFrame(predicted_answers)
        predicted_answers[1] =predicted_answers[1].rank(method= 'first',ascending =False)
        #predicted_answers = dict(predicted_answers)

            
        for index, row in predicted_answers.iterrows():
            if e == row[0]:
                 return int(row[1])
        return -1


    def get_hits(self,answers, list_answers: List[str], gold_answers: List[str], query: Tuple[str, str]) \
            -> Tuple[float, float, float, float, float]:
        hits_1 = 0.0
        hits_3 = 0.0
        hits_5 = 0.0
        hits_10 = 0.0
        rr = 0.0
        (e1, r) = query
        all_gold_answers = self.args.all_kg_map[(e1, r)]
        for gold_answer in gold_answers:
            # remove all other gold answers from prediction
            filtered_answers = []
            for pred, score in answers:
                if pred in all_gold_answers and pred != gold_answer:
                    continue
                else:
                    filtered_answers.append((pred,score))
            #print(filtered_answers)
            self.top_query_preds[(e1, r, gold_answer)] = filtered_answers[:10]
            rank = ProbCBR.get_rank_in_list(gold_answer, filtered_answers)
            if rank > 0:
                if rank <= 10:
                    hits_10 += 1
                    if rank <= 5:
                        hits_5 += 1
                        if rank <= 3:
                            hits_3 += 1
                            if rank <= 1:
                                hits_1 += 1
                rr += 1.0 / rank
        return hits_10, hits_5, hits_3, hits_1, rr

    @staticmethod
    def get_accuracy(gold_answers: List[str], list_answers: List[str]) -> List[float]:
        all_acc = []
        for gold_ans in gold_answers:
            if gold_ans in list_answers:
                all_acc.append(1.0)
            else:
                all_acc.append(0.0)
        return all_acc
    
    
    def execute_program_ents(self, ent,  program, max_branch=20):
        q = deque()
        q1 = deque()
        solutions = defaultdict(list)
        solutions1 = defaultdict(list)
        q.append((ent, 0, []))
        q1.append((ent, 0, []))
        while len(q1):
            e1, depth, path = q1.popleft()
            if depth == len(program):
                #solutions[e1].append(path + [(self.entity_vocab[e1],
                #                              len(self.rel_vocab))])
                solutions1[e1].append(path + [(e1,
                                              len(rel))])
                continue
            rel = program[depth]
            next_entities = self.full_map[e1, rel]
            if len(next_entities) > max_branch:
                next_entities = np.random.choice(next_entities, max_branch,
                                                 replace=False)
            depth += 1
            for e2 in next_entities:
                #q.append((e2, depth, path + [(self.entity_vocab[e1],
                #                              self.rel_vocab[rel])]))
                q1.append((e2, depth, path + [(e1,rel)]))
        return solutions1
  

    def get_entity_programs(self, e1, programs):
        programs_to_entity = defaultdict(list)
        for p in programs:
            for ent, programs in self.execute_program_ents(e1, p).items():
                programs_to_entity[ent].extend(programs)
        return programs_to_entity

    def do_symbolic_case_based_reasoning(self):
        num_programs = []
        num_answers = []
        all_acc = []
        all_acc_top = []
        non_zero_ctr = 0
        hits_10, hits_5, hits_3, hits_1, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        per_relation_scores = {}  # map of performance per relation
        per_relation_query_count = {}
        total_examples = 0
        learnt_programs = defaultdict(lambda: defaultdict(int))  # for each query relation, a map of programs to count
        all_data =[]
        for ex_ctr, ((e1, r), e2_list) in enumerate(tqdm(self.eval_map.items())):
            #logger.info("Executing query {}".format(ex_ctr))
            # if e2_list is in train list then remove them
            # Normally, this shouldn't happen at all, but this happens for Nell-995.
            
            query_data = {
                'Drug': (e1,MRN_nodes[MRN_nodes['id']==e1]['name'].values[0]),
                'relation': r,
                'True Indication (supported by Drug Central)': e2_list
            }
            orig_train_e2_list = self.train_map[(e1, r)]
            temp_train_e2_list = []
            for e2 in orig_train_e2_list:
                if e2 in e2_list:
                    continue
                temp_train_e2_list.append(e2)
            self.train_map[(e1, r)] = temp_train_e2_list
            # also remove (e2, r^-1, e1)
            r_inv = get_inv_relation(r, self.args.dataset_name)
            temp_map = {}  # map from (e2, r_inv) -> outgoing nodes
            for e2 in e2_list:
                temp_map[(e2, r_inv)] = self.train_map[e2, r_inv]
                temp_list = []
                for e1_dash in self.train_map[e2, r_inv]:
                    if e1_dash == e1:
                        continue
                    else:
                        temp_list.append(e1_dash)
                self.train_map[e2, r_inv] = temp_list

            total_examples += len(e2_list)
            if e1 not in self.entity_vocab:
                all_acc += [0.0] * len(e2_list)
                # put it back
                self.train_map[(e1, r)] = orig_train_e2_list
                for e2 in e2_list:
                    self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
                continue  # this entity was not seen during train; skip?
            self.c = self.args.cluster_assignments[self.entity_vocab[e1]]
            num_nn_for_r = self.args.k_adj if self.per_relation_config is None else self.per_relation_config[r]["k_adj"]
            all_programs = self.get_programs_from_nearest_neighbors(e1, r, self.get_nearest_neighbor_inner_product,
                                                                    num_nn=num_nn_for_r)
            for p in all_programs:
                if p[0] == r:
                    continue
                if r not in learnt_programs:
                    learnt_programs[r] = {}
                p = tuple(p)
                if p not in learnt_programs[r]:
                    learnt_programs[r][p] = 0
                learnt_programs[r][p] += 1
                
            #out_file_name = os.path.join(args.data_dir, "learnt_programs.json")
            #with open(out_file_name, "wb") as fout:
                #pickle.dump(learnt_programs, fout)
            

            # filter the program if it is equal to the query relation
            temp = []
            for p in all_programs:
                if len(p) == 1 and p[0] == r:
                    continue
                temp.append(p)
            all_programs = temp

            if len(all_programs) > 0:
                non_zero_ctr += len(e2_list)

            all_uniq_programs, sorted_programs_with_score = self.rank_programs(all_programs, r)
            #query_data['programs'] = sorted_programs_with_score
            
            #for u_p in all_uniq_programs:
             #   learnt_programs[r][u_p] += 1
            
            num_programs.append(len(all_uniq_programs))
            # Now execute the program
            answers, not_executed_programs = self.execute_programs(e1, r, all_uniq_programs, max_branch=self.args.max_branch)
            #print(answers)
            #query_data['answers']=answers
            aggr_type1_for_r = self.args.aggr_type1 if self.per_relation_config is None \
                else self.per_relation_config[r]["aggr_type1"]
            aggr_type2_for_r = self.args.aggr_type2 if self.per_relation_config is None \
                else self.per_relation_config[r]["aggr_type2"]
            answers = self.rank_answers(answers,
                                        aggr_type1_for_r,
                                        aggr_type2_for_r)
               
            
            entity_paths = self.get_entity_programs(e1, all_uniq_programs)
            num = int(self.ans_num)
            predicted_answers = [e for e, score in answers][0:num]
            #query_data['Predicted drugs'] = predicted_answers

            #query_data['entity_paths'] = dict(entity_paths)
            
          
            predicted_answers_entpaths = [(e,MRN_nodes[MRN_nodes['id']==e]['name'].values[0], float(score),entity_paths[e]) for e, score in answers if entity_paths[e] != []]
            
            #query_data['entity_paths'] = [entity_paths[e] for e, score in predicted_answers]
                
            #print(self.ans_num)
            
            query_data['Predicted drugs with paths'] = predicted_answers_entpaths[0:num]  # to save as json
            

            # put it back
            self.train_map[(e1, r)] = orig_train_e2_list
            for e2 in e2_list:
                self.train_map[(e2, r_inv)] = temp_map[(e2, r_inv)]
            all_data.append(query_data)
       
        if self.args.dump_paths:
            out_file_name = os.path.join(self.args.data_dir, "data.json")
            fout = open(out_file_name, "w")
            fout.write(json.dumps(all_data[0], indent=4))
            fout.close()

        query_data_json = json.dumps(all_data[0], sort_keys = True, indent = 4, separators = (',', ': '))
        return query_data['Predicted drugs with paths'], query_data_json
            
    