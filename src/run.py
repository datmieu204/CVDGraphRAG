import os
from camel.storages import Neo4jGraph
from dataloader import load_high
import argparse
from creat_graph_with_description import creat_metagraph_with_description
from summerize import process_chunks
from retrieve import seq_ret
from utils import *
from nano_graphrag import GraphRAG, QueryParam

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-simple', action='store_true')
parser.add_argument('-construct_graph', action='store_true')
parser.add_argument('-inference',  action='store_true')
parser.add_argument('-grained_chunk',  action='store_true')
parser.add_argument('-trinity', action='store_true')
parser.add_argument('-trinity_gid1', type=str)
parser.add_argument('-trinity_gid2', type=str)
parser.add_argument('-ingraphmerge',  action='store_true')
parser.add_argument('-crossgraphmerge', action='store_true')
parser.add_argument('-dataset', type=str, default='mimic_ex')
parser.add_argument('-data_path', type=str, default='./dataset_test')
parser.add_argument('-test_data_path', type=str, default='./dataset_ex/report_0.txt')
args = parser.parse_args()

if args.simple:
    graph_func = GraphRAG(working_dir="./nanotest")

    with open("./dataset_ex/report_0.txt") as f:
        graph_func.insert(f.read())

    print(graph_func.query("What is the main symptom of the patient?", param=QueryParam(mode="local")))

else:

    url=os.getenv("NEO4J_URL")
    username=os.getenv("NEO4J_USERNAME")
    password=os.getenv("NEO4J_PASSWORD")

    # Set Neo4j instance
    n4j = Neo4jGraph(
        url=url,
        username=username,
        password=password
    )

    if args.construct_graph: 
        if args.dataset == 'mimic_ex':
            files = [file for file in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, file))]
            
            for file_name in files:
                file_path = os.path.join(args.data_path, file_name)
                content = load_high(file_path)
                gid = str_uuid()
                n4j = creat_metagraph_with_description(args, content, gid, n4j)

                if args.trinity:
                    link_context(n4j, args.trinity_gid1)
            if args.crossgraphmerge:
                merge_similar_nodes(n4j, None)

    if args.inference:
        from dedicated_key_manager import create_dedicated_client
        
        question = load_high("./prompt.txt")
        
        inference_client = create_dedicated_client(task_id="inference_main")
        sum = process_chunks(question, client=inference_client)
        gid = seq_ret(n4j, sum)
        
        if gid is None:
            print("\nCannot perform inference - database is empty or no valid results found.")
            print("Run the following command first to build the graph:")
            print("python run.py -dataset mimic_ex -data_path ../data/mimic_ex -construct_graph")
        else:
            response = get_response(n4j, gid, question)
            print(response)
