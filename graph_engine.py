import networkx as nx
import pandas as pd
import numpy as np

# Global graph object
FRAUD_GRAPH = nx.DiGraph()
KNOWN_FRAUD_NODES = set() # Placeholder for known fraud/mule accounts

def update_graph(txn: dict):
    """Adds a transaction edge to the global fraud graph."""
    sender = txn['sender_id']
    receiver = txn['receiver_id']
    amount = txn['amount']
    timestamp = txn['timestamp']

    # Add nodes (users) if they don't exist
    if sender not in FRAUD_GRAPH:
        FRAUD_GRAPH.add_node(sender, type='user')
    if receiver not in FRAUD_GRAPH:
        FRAUD_GRAPH.add_node(receiver, type='user')

    # Add edge (transaction)
    # Weight can be inverse of amount or amount itself, we use amount here
    FRAUD_GRAPH.add_edge(sender, receiver, 
                         weight=amount, 
                         timestamp=timestamp, 
                         id=txn['transaction_id'])
    
    # Basic mule account heuristic (for simulation purposes)
    if txn.get('is_high_value_simulated', False) and receiver.startswith('NEW_U'):
        KNOWN_FRAUD_NODES.add(receiver)
        
    if txn.get('is_new_receiver', False) and txn.get('amount') > 50000:
        KNOWN_FRAUD_NODES.add(receiver)

def get_graph_score(txn: dict) -> tuple:
    """Calculates the graph-based fraud score (0-1) and reasons."""
    
    graph_score = 0.0
    reasons = []
    
    sender = txn['sender_id']
    receiver = txn['receiver_id']

    # Must have at least 10 nodes for centrality measures to be meaningful
    if FRAUD_GRAPH.number_of_nodes() < 10:
        return 0.0, ["Graph building in progress"] 

    try:
        # 1. Degree Centrality (High degree can indicate a mule account)
        sender_degree = FRAUD_GRAPH.degree(sender)
        receiver_degree = FRAUD_GRAPH.degree(receiver)
        
        # Simple heuristic: if a node has > 100 incoming/outgoing edges, it's suspicious
        if sender_degree > 100 or receiver_degree > 100:
            graph_score += 0.4
            reasons.append(f"High Degree Centrality (S:{sender_degree}, R:{receiver_degree}) (+0.4)")

        # 2. Closeness to Known Fraud Nodes (Proximity to "Fraud Rings")
        closeness_score = 0.0
        if KNOWN_FRAUD_NODES:
            # Check shortest path distance (undirected) to any known fraud node
            min_dist = np.inf
            for fraud_node in KNOWN_FRAUD_NODES:
                try:
                    if nx.has_path(FRAUD_GRAPH.to_undirected(), sender, fraud_node):
                        dist = nx.shortest_path_length(FRAUD_GRAPH.to_undirected(), sender, fraud_node)
                        min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    pass
            
            # Simple scoring: dist 1/2 is high risk. dist 3/4 is medium.
            if min_dist == 1:
                closeness_score = 0.4 # Directly connected to a mule/fraud node
            elif min_dist == 2:
                closeness_score = 0.2
            
            if closeness_score > 0:
                graph_score += closeness_score
                reasons.append(f"Close to Known Fraud Node (Dist: {min_dist}) (+{closeness_score})")

    except nx.NetworkXError as e:
        # Node not in graph yet (unlikely if update_graph is run first)
        return 0.0, [f"Graph analysis error: {e}"]

    # 3. Simple high in-degree to a new receiver (Potential Mule Acquisition)
    if receiver.startswith('NEW_U') and FRAUD_GRAPH.in_degree(receiver) > 5:
        graph_score += 0.3
        reasons.append("New Receiver with High In-Degree (+0.3)")


    graph_score = min(graph_score, 1.0)
    
    return graph_score, reasons

if __name__ == '__main__':
    # Test stub
    print("Graph Engine loaded.")