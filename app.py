from flask import Flask, render_template, request
import pickle
from pr_cbr import ProbCBR
from argparse import Namespace
import numpy as np
import os
import json
import pickle
import  sys
import torch
from ext.data.data_utils import (
    create_vocab,
    load_data,
    load_vocab,
    get_unique_entities,
    read_graph,
    get_entities_group_by_relation,
    load_data_all_triples,
)
from ext.preprocessing import combine_path_splits
from typing import *
import logging
import json

# import wandb
torch.cuda.empty_cache()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s \t %(message)s]", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


# Create flask app
flask_app = Flask(__name__)



device = None



def get_model(args):
    dataset_name = args.dataset_name
    logger.info("==========={}============".format(dataset_name))
    data_dir = os.path.join(args.data_dir, "data", dataset_name)
    subgraph_dir = os.path.join(args.data_dir, "subgraphs", dataset_name,
                                "paths_{}".format(args.num_paths_around_entities))
    kg_file = os.path.join(data_dir, "full_graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                   "graph.txt")
    if args.small:
        if args.input_file_name is not None:
            if args.test:
                args.test_file = os.path.join(data_dir, "inputs", "test", args.input_file_name + ".small")
                args.dev_file = os.path.join(data_dir, "dev.txt.small")
            else:
                args.dev_file = os.path.join(data_dir, "inputs", "valid", args.input_file_name + ".small")
                args.test_file = os.path.join(data_dir, "test.txt")
        elif args.specific_rel is not None:
            args.dev_file = os.path.join(data_dir, f"dev.{args.specific_rel}.txt.small")
            args.test_file = os.path.join(data_dir, "test.txt")
        else:
            args.dev_file = os.path.join(data_dir, "dev.txt.small")
            args.test_file = os.path.join(data_dir, "test.txt")
    else:
        if args.input_file_name is not None:
            if args.test:
                args.test_file = os.path.join(data_dir, "inputs", "test", args.input_file_name)
                args.dev_file = os.path.join(data_dir, "dev.txt")
            else:
                args.dev_file = os.path.join(data_dir, "inputs", "valid", args.input_file_name)
                args.test_file = os.path.join(data_dir, "test.txt")
        elif args.specific_rel is not None:
            args.dev_file = os.path.join(data_dir, f"dev.{args.specific_rel}.txt")
            args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
                else os.path.join(data_dir, args.test_file_name)
        else:
            args.dev_file = os.path.join(data_dir, "dev.txt")
            args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
                else os.path.join(data_dir, args.test_file_name)

    args.train_file = os.path.join(data_dir, "graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                      "train.txt")
    logger.info("Loading train map")
    train_map = load_data(kg_file)
    full_map = load_data(kg_file)
    logger.info("Loading dev map")
    dev_map = load_data(args.dev_file)
    logger.info("Loading test map")
    test_map = load_data(args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file
    rel_ent_map = get_entities_group_by_relation(args.train_file)

    #logger.info("=========Config:============")
    #logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    if args.per_relation_config_file is not None and os.path.exists(args.per_relation_config_file):
        per_relation_config = json.load(open(args.per_relation_config_file))
        #logger.info("=========Per Relation Config:============")
        #logger.info(json.dumps(per_relation_config, indent=1, sort_keys=True))
    else:
        per_relation_config = None
    logger.info("Loading vocabs...")
    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab = load_vocab(data_dir)
    # making these part of args for easier access #hack
    # args.entity_vocab = entity_vocab
    # args.rel_vocab = rel_vocab
    # args.rev_entity_vocab = rev_entity_vocab
    # args.rev_rel_vocab = rev_rel_vocab
    # args.train_map = train_map
    # args.dev_map = dev_map
    # args.test_map = test_map

    # logger.info("Loading combined train/dev/test map for filtered eval")
    all_kg_map = load_data_all_triples(train_file, os.path.join(data_dir, 'dev.txt'),
                                        os.path.join(data_dir, 'test.txt'))
    args.all_kg_map = all_kg_map

    ########### Load all paths ###########
    file_prefix = "paths_{}_path_len_{}_".format(args.num_paths_around_entities, args.max_path_len)
    all_paths = combine_path_splits(subgraph_dir, file_prefix=file_prefix)

    return ProbCBR(args, train_map, full_map, eval_map, entity_vocab, rev_entity_vocab, rel_vocab,
                             rev_rel_vocab, eval_vocab, eval_rev_vocab, all_paths, rel_ent_map, per_relation_config, args.ans_num)
 

args = Namespace(
    dataset_name = "MIND_CtD",
    data_dir = "cbr-akbc-data",
    expt_dir="../prob-expts/",
    subgraph_file_name= "paths_1000.pkl",
    per_relation_config_file = None,
    test = True,
    test_file_name="test.txt",
    input_file_name = None,
    use_path_counts = 1,
    linkage=0.0,
    k_adj=10,
    max_num_programs = 1000,
    output_dir = "results/",
    output_per_relation_scores = True,
    print_paths = True,
    num_paths_around_entities = 1000,
    max_path_len=4,
    prevent_loops=1,
    max_branch = 100,
    aggr_type1 = "none",
    aggr_type2 = "sum",
    use_only_precision_scores = 0,
    specific_rel = None,
    dump_paths = True,
    small =False,
    cheat_neighbors = 0

) 
dataset_name = args.dataset_name  
data_dir = os.path.join(args.data_dir, "data", dataset_name)
kg_file = os.path.join(data_dir, "full_graph.txt") if args.dataset_name == "nell" else os.path.join(data_dir,"graph.txt")
dev_file = os.path.join(data_dir, "dev.txt")
#test_file = os.path.join(data_dir, "test.txt")
train_file = os.path.join(data_dir, "train.txt")
subgraph_dir = os.path.join(args.data_dir, "subgraphs", dataset_name,"paths_{}".format(args.num_paths_around_entities))                                                                                               

train_map = load_data(kg_file)
full_map = load_data(kg_file)
#logger.info("Loading dev map")
dev_map = load_data(dev_file)
#logger.info("Loading test map")
#test_map = load_data(test_file)
eval_map = dev_map
entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab = load_vocab(data_dir)
rel_ent_map = get_entities_group_by_relation(train_file)
if args.per_relation_config_file is not None and os.path.exists(args.per_relation_config_file):
    per_relation_config = json.load(open(args.per_relation_config_file))
#logger.info("=========Per Relation Config:============")
    #     #logger.info(json.dumps(per_relation_config, indent=1, sort_keys=True))
else:
    per_relation_config = None

########### entity sim ###########
if os.path.exists(os.path.join(args.data_dir, "data", args.dataset_name, "ent_sim.pkl")):
    with open(os.path.join(args.data_dir, "data", args.dataset_name, "ent_sim.pkl"), "rb") as fin:
        sim_and_ind = pickle.load(fin)
        sim = sim_and_ind["sim"]
        arg_sim = sim_and_ind["arg_sim"]
else:
    logger.info(
        "Entity similarity matrix not found at {}. Please run the preprocessing script first to generate this matrix...".format(
            os.path.join(args.data_dir, "data", args.dataset_name, "ent_sim.pkl")))
    sys.exit(1)


########### cluster entities ###########
dir_name = os.path.join(args.data_dir, "data", args.dataset_name, "linkage={}".format(args.linkage))
cluster_file_name = os.path.join(dir_name, "cluster_assignments.pkl")
if os.path.exists(cluster_file_name):
    with open(cluster_file_name, "rb") as fin:
        args.cluster_assignments = pickle.load(fin)
else:
    logger.info(
        "Clustering file not found at {}. Please run the preprocessing script first".format(cluster_file_name))
    sys.exit(1)

########### load prior maps ###########
path_prior_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "prior_maps",
                                         "path_{}".format(args.num_paths_around_entities), "path_prior_map.pkl")
logger.info("Loading path prior weights")
if os.path.exists(path_prior_map_filenm):
    with open(path_prior_map_filenm, "rb") as fin:
        args.path_prior_map_per_relation = pickle.load(fin)
else:
    logger.info(
        "Path prior files not found at {}. Please run the preprocessing script".format(path_prior_map_filenm))

########### load prior maps (fall-back) ###########
linkage_bck = args.linkage
args.linkage = 0.0
bck_dir_name = os.path.join(data_dir, "linkage={}".format(args.linkage), "prior_maps",
                                "path_{}".format(args.num_paths_around_entities))
path_prior_map_filenm_fallback = os.path.join(bck_dir_name, "path_prior_map.pkl")
if os.path.exists(bck_dir_name):
    logger.info("Loading fall-back path prior weights")
    with open(path_prior_map_filenm_fallback, "rb") as fin:
        args.path_prior_map_per_relation_fallback = pickle.load(fin)
else:
    logger.info("Fall-back path prior weights not found at {}. Please run the preprocessing script".format(
        path_prior_map_filenm_fallback))
args.linkage = linkage_bck

########### load precision maps ###########
precision_map_filenm = os.path.join(data_dir, "linkage={}".format(args.linkage), "precision_maps",
                                        "path_{}".format(args.num_paths_around_entities), "precision_map.pkl")
logger.info("Loading precision map")
if os.path.exists(precision_map_filenm):
    with open(precision_map_filenm, "rb") as fin:
        args.precision_map = pickle.load(fin)
else:
    logger.info(
            "Path precision files not found at {}. Please run the preprocessing script".format(precision_map_filenm))

########### load precision maps (fall-back) ###########
linkage_bck = args.linkage
args.linkage = 0.0
precision_map_filenm_fallback = os.path.join(data_dir, "linkage={}".format(args.linkage), "precision_maps",
                                                 "path_{}".format(args.num_paths_around_entities), "precision_map.pkl")
logger.info("Loading fall-back precision map")
if os.path.exists(precision_map_filenm_fallback):
    with open(precision_map_filenm_fallback, "rb") as fin:
        args.precision_map_fallback = pickle.load(fin)
else:
    logger.info("Path precision fall-back files not found at {}. Please run the preprocessing script".format(
        precision_map_filenm_fallback))
args.linkage = linkage_bck


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])


def predict():
    float_features = [val for key, val in request.form.items()]
    args.ans_num = float_features[1]
    with open(os.path.join(args.data_dir, "data/MIND_CtD/test.txt"), 'w') as f:
        f.write(float_features[0]+ "\t" + "indication_inv" + "\t" + "CHEBI:141521")
         #f.write(float_features[0]+ "\t" + "indication_inv")

    dataset_name = args.dataset_name
    #logger.info("==========={}============".format(dataset_name))
    data_dir = os.path.join(args.data_dir, "data", dataset_name)
    subgraph_dir = os.path.join(args.data_dir, "subgraphs", dataset_name,
                                "paths_{}".format(args.num_paths_around_entities))
    kg_file = os.path.join(data_dir, "full_graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                  "graph.txt")
    if args.small:
        if args.input_file_name is not None:
            if args.test:
                args.test_file = os.path.join(data_dir, "inputs", "test", args.input_file_name + ".small")
                args.dev_file = os.path.join(data_dir, "dev.txt.small")
            else:
                args.dev_file = os.path.join(data_dir, "inputs", "valid", args.input_file_name + ".small")
                args.test_file = os.path.join(data_dir, "test.txt")
        elif args.specific_rel is not None:
            args.dev_file = os.path.join(data_dir, f"dev.{args.specific_rel}.txt.small")
            args.test_file = os.path.join(data_dir, "test.txt")
        else:
            args.dev_file = os.path.join(data_dir, "dev.txt.small")
            args.test_file = os.path.join(data_dir, "test.txt")
    else:
        if args.input_file_name is not None:
            if args.test:
                args.test_file = os.path.join(data_dir, "inputs", "test", args.input_file_name)
                args.dev_file = os.path.join(data_dir, "dev.txt")
            else:
                args.dev_file = os.path.join(data_dir, "inputs", "valid", args.input_file_name)
                args.test_file = os.path.join(data_dir, "test.txt")
        elif args.specific_rel is not None:
            args.dev_file = os.path.join(data_dir, f"dev.{args.specific_rel}.txt")
            args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
                else os.path.join(data_dir, args.test_file_name)
        else:
            args.dev_file = os.path.join(data_dir, "dev.txt")
            args.test_file = os.path.join(data_dir, "test.txt") if not args.test_file_name \
                else os.path.join(data_dir, args.test_file_name)

    args.train_file = os.path.join(data_dir, "graph.txt") if dataset_name == "nell" else os.path.join(data_dir,
                                                                                                      "train.txt")

    test_map = load_data(args.test_file)
    eval_map = dev_map
    eval_file = args.dev_file
    if args.test:
        eval_map = test_map
        eval_file = args.test_file
    rel_ent_map = get_entities_group_by_relation(args.train_file)

    #logger.info("=========Config:============")
    #logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    if args.per_relation_config_file is not None and os.path.exists(args.per_relation_config_file):
        per_relation_config = json.load(open(args.per_relation_config_file))
        #logger.info("=========Per Relation Config:============")
        #logger.info(json.dumps(per_relation_config, indent=1, sort_keys=True))
    else:
        per_relation_config = None
    #logger.info("Loading vocabs...")
    entity_vocab, rev_entity_vocab, rel_vocab, rev_rel_vocab, eval_vocab, eval_rev_vocab = load_vocab(data_dir)
    #making these part of args for easier access #hack
    args.entity_vocab = entity_vocab
    args.rel_vocab = rel_vocab
    args.rev_entity_vocab = rev_entity_vocab
    args.rev_rel_vocab = rev_rel_vocab
    args.train_map = train_map
    args.dev_map = dev_map
    args.test_map = test_map
    args.cheat_neighbors = 0

    #logger.info("Loading combined train/dev/test map for filtered eval")
    all_kg_map = load_data_all_triples(args.train_file, os.path.join(data_dir, 'dev.txt'),
                                       os.path.join(data_dir, 'test.txt'))
    args.all_kg_map = all_kg_map

    ########### Load all paths ###########
    file_prefix = "paths_{}_path_len_{}_".format(args.num_paths_around_entities, args.max_path_len)
    all_paths = combine_path_splits(subgraph_dir, file_prefix=file_prefix)

    prob_cbr_agent = get_model(args)

    assert arg_sim is not None
    prob_cbr_agent.set_nearest_neighbor_1_hop(arg_sim)

    output, query_data = prob_cbr_agent.do_symbolic_case_based_reasoning()
    return render_template("index.html", prediction_text = "Top ten predicted answers: {}".format(output), response=query_data)
    #return output

if __name__ == "__main__":
    flask_app.run(host=os.getenv('IP', '0.0.0.0'), port=int(os.getenv('PORT', 4442)),debug=True)
