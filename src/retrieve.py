from utils import *

sys_p = """
Assess the similarity of the two provided summaries and return a rating from these options: 'very similar', 'similar', 'general', 'not similar', 'totally not similar'. Provide only the rating.
"""

from logger_ import get_logger

logger = get_logger("retrieve", log_file="logs/retrieve.log")

def seq_ret(n4j, sumq):
    rating_list = []
    sumk = []
    gids = []
    sum_query = """
        MATCH (s:Summary)
        RETURN s.content, s.gid
        """
    res = n4j.query(sum_query)
    
    # Check if database has any Summary nodes
    if not res:
        logger.error("❌ Error: No Summary nodes found in Neo4j database.")
        logger.info("💡 Please run with -construct_graph first to build the knowledge graph.")
        return None
    
    for r in res:
        sumk.append(r['s.content'])
        gids.append(r['s.gid'])
    
    for sk in sumk:
        sk = sk[0]
        rate = call_llm(sys_p, "The two summaries for comparison are: \n Summary 1: " + sk + "\n Summary 2: " + sumq[0])
        if "totally not similar" in rate:
            rating_list.append(0)
        elif "not similar" in rate:
            rating_list.append(1)
        elif "general" in rate:
            rating_list.append(2)
        elif "very similar" in rate:
            rating_list.append(4)
        elif "similar" in rate:
            rating_list.append(3)
        else:
            logger.warning("llm returns no relevant rate")
            rating_list.append(-1)

    ind = find_index_of_largest(rating_list)
    
    # Handle case when no valid index found
    if ind == -1:
        logger.error("❌ Error: Could not find any valid ratings.")
        return None
    
    gid = gids[ind]

    return gid
