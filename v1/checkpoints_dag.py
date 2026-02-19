

# from Dynamic Hierarchical Learning paper

# (a dynamic DAG as the self-referential levels.... just as a proof of concept)



# also, from a minimal dag-like blockchain implementation








# Checkpoint: Agent metadata, Cog. Arch. (with Short-Term Memory storage), Optimizer State(?), etc







# from https://github.com/nirel1/Merkle-DAG-Blockchain/blob/main/blockchain/dag_blockchain.py










from block import Block
from blockchain_utils import BlockchainUtils
from account_model import AccountModel
from proof_of_stake import ProofOfStake
from pydag import DAG
import block
from matplotlib import pyplot as plt

# for visualize
import itertools as itrt
from collections import deque
from ordered_set import OrderedSet
from typing import Iterator, AbstractSet, Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np


class DAGBlockchain:
    def __init__(self, dimensions, gnode_id, genesis_forger, cluster_id=-1):
        dag = DAG()
        g_block = block.Block.genesis(gnode_id, genesis_forger)
        g_block.cluster_id = cluster_id
        dag.add_node(g_block)
        self.blocks = [g_block]
        self.right_size = 0
        self.left_size = 0
        self.size = 0
        self.account_model = AccountModel()
        #self.pos = ProofOfStake()
        self.chain_id = 0
        self.gnode_id = gnode_id
        self.dag = dag
        self.genesis_forger = genesis_forger
        self.dimensions = dimensions

    def get_coord_center(self, cluster_num):
        coord_center = [50, 50]
        if cluster_num == -1:
            return coord_center

        if cluster_num % 4 == 0:
            coord_center[0] = 87.5
        elif cluster_num % 4 == 3:
            coord_center[0] = 62.5
        elif cluster_num % 4 == 2:
            coord_center[0] = 37.5
        else:
            coord_center[0] = 12.5

        if cluster_num < 5:
            coord_center[1] = 12.5
        elif cluster_num < 9:
            coord_center[1] = 37.5
        elif cluster_num < 13:
            coord_center[1] = 62.5
        else:
            coord_center[1] = 87.5
        return coord_center


    def add_block(self, new_block):
        if new_block.node_id not in self.dag.graph:
            self.size += 1
            self.execute_transactions(new_block.transactions)
            self.blocks.append(new_block)
            self.dag.add_node(new_block)
            if self.dimensions == 4:
                center = self.get_coord_center(new_block.cluster_id)
                self.dag.add_edge(new_block.node_id, self.dag.leaf_selection(new_block, self.gnode_id, center[0], center[1]))
            else:
                self.dag.add_edge(new_block.node_id, self.dag.leaf_selection2(new_block))
        else:
            print('block add failed')

    def add_blocks(self, new_blocks):
        for block in new_blocks:
            self.add_block(block)

    def slow_merge(self, blockchain2):
        for block in blockchain2.blocks:
            self.add_block(block)

    def merge(self, blockchain2):
        for first_nodes in blockchain2.dag.predecessors(blockchain2.dag.all_leaves()[0]):
            blockchain2.dag.graph[first_nodes].edges.remove(blockchain2.gnode_id)
            self.dag.add_node_chain(first_nodes, blockchain2.dag.graph)
            self.dag.add_edge(
                first_nodes,
                self.dag.leaf_selection(blockchain2.dag.graph[first_nodes].block, self.gnode_id, 50, 50))
            # blockchain2.dag.delete_node(blockchain2.dag.graph[blockchain2.gnode_id].block)

    def to_json(self):
        data = {}
        json_blocks = []
        for block in self.blocks:
            json_blocks.append(block.to_json())
        data['blocks'] = json_blocks
        return data

    # NOTE: Might not need blockcount
    def block_count_valid(self, block):
        if self.blocks[-1].block_count == block.block_count - 1:
            return True
        else:
            return False

    def parent_block_hash_valid(self, block):
        latest_blockchain_block_hash = BlockchainUtils.hash(
            self.blocks[-1].payload()).hexdigest()
        if latest_blockchain_block_hash == block.last_hash:
            return True
        else:
            return False

    def get_covered_transaction_set(self, transactions):
        covered_transactions = []
        for transaction in transactions:
            if self.transaction_covered(transaction):
                covered_transactions.append(transaction)
            else:
                print('transaction is not covered by sender')
        return covered_transactions

    def transaction_covered(self, transaction):
        if transaction.tr_type == 'EXCHANGE':
            return True
        sender_balance = self.account_model.get_balance(
            transaction.sender_public_key)
        if sender_balance >= transaction.amount:
            return True
        else:
            return False

    def execute_transactions(self, transactions):
        for transaction in transactions:
            self.execute_transaction(transaction)

    def execute_transaction(self, transaction):
        if transaction.tr_type == 'STAKE':
            sender = transaction.sender_public_key
            receiver = transaction.receiver_public_key
            if sender == receiver:
                amount = transaction.amount
                self.pos.update(sender, amount)
                self.account_model.update_balance(sender, -amount)
        else:
            sender = transaction.sender_public_key
            receiver = transaction.receiver_public_key
            amount = transaction.amount
            self.account_model.update_balance(sender, -amount)
            self.account_model.update_balance(receiver, amount)

    def next_forger(self):
        parent_block_hash = BlockchainUtils.hash(
            self.blocks[-1].payload()).hexdigest()
        next_forger = self.pos.forger(parent_block_hash)
        return next_forger

    def create_block(self, transactions_from_pool, forger_wallet, node_id):
        covered_transactions = self.get_covered_transaction_set(
            transactions_from_pool)
        self.execute_transactions(covered_transactions)
        new_block = forger_wallet.create_block(covered_transactions, node_id)
        return_data = self.blocks.add(new_block)
        parent = return_data[1]
        traversed_kdnodes = return_data[2]
        return new_block, parent, traversed_kdnodes

    def transaction_exists(self, transaction):
        for node in self.blocks:
            for block_transaction in node.transactions:
                if transaction.id.equals(block_transaction.id):
                    return True
        return False

    def forger_valid(self, block):
        forger_public_key = self.pos.forger(block.last_hash)
        proposed_block_forger = block.forger
        if forger_public_key == proposed_block_forger:
            return True
        else:
            return False

    def transactions_valid(self, transactions):
        covered_transactions = self.get_covered_transaction_set(transactions)
        if len(covered_transactions) == len(transactions):
            return True
        return False

    def merkle_root(self):
        return self.blocks.subtree_hash

    # TODO: loop through bc to add each node to self instead of the entire bc tree
    # somewhat done ?
    # def merge(self, bc):
    #     for b in bc.levelorder():
    #         self.blocks.add_node(b)

    def visualize(self, text):
        visual = nx.DiGraph()
        for element in self.dag.graph.items():
            node_id = element[0]
            visual.add_node(node_id)
            for edge in self.dag.graph[node_id].edges:
                visual.add_edge(node_id, edge)
        nx.draw_networkx(visual, arrows=True, node_color='white')
        if text != "":
            plt.savefig(text)
















# from https://github.com/nirel1/Merkle-DAG-Blockchain/blob/main/blockchain/block.py





import time
import copy
from blockchain_utils import BlockchainUtils as BU

# Block class temporarily inherits dict to make JSON serialization easy.
# Should be changed to be more robust at a later time.
class Block(dict):
    # def __init__(self, transactions, parent_hash, x, y, forger, block_count):
    def __init__(self, transactions, node_id, x, y, forger, t=time.time(), parent_hash=None, signature=''):
        dict.__init__(self)
        # self.block_count = block
        self.coords = [node_id, x, y, t]
        self.transactions = transactions
        self.parent_hash = parent_hash
        self.forger = forger
        self.signature = signature
        self.node_id = node_id
        self.cluster_id = -1

    def __getitem__(self, item):
        return self.coords[item]

    # block[1] -> return self.coors[1]
    def __setitem__(self, key, value):
        self.coords[key] = value

    def __len__(self):
        return len(self.coords)

    def __repr__(self):
        # return f'Block({self.coords}, {self.forger}, {self.parent_hash}, {self.transactions})'
        # return f'Block({self.coords}, {self.transactions},\n' \
        #        f'{self.parent_hash},\n' \
        #        f'{self.forger},\n' \
        #        f'{self.signature})'
        return f'Block({self.coords})'

    def __hash__(self):
        return hash((self.cluster_id, self.node_id, self.coords[1], self.coords[2], self.coords[3]))

    @staticmethod
    def genesis(genesis_node_id, forger):
        genesis_block = Block([], genesis_node_id, x=0, y=0, forger=forger, parent_hash='0')
        # genesis_block.timestamp = 0
        return genesis_block

    def to_json(self):
        j_data = {'coordinates': self.coords,
                  'signature': self.signature,
                  'parent_hash': self.parent_hash}

        json_transactions = []
        for transaction in self.transactions:
            json_transactions.append(transaction.to_json())
        j_data['transactions'] = json_transactions

        return j_data

    def payload(self):
        json_representation = copy.deepcopy(self.to_json())
        json_representation['signature'] = ''
        return json_representation

    def sign(self, signature):
        self.signature = signature


















    
    # from https://github.com/nirel1/Merkle-DAG-Blockchain/blob/main/blockchain/pydag/__init__.py










    from copy import copy, deepcopy
from collections import deque
from blockchain_utils import BlockchainUtils as BU
from . import six_subset as six

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class DAGValidationError(Exception):
    pass


class DAGNode(object):
    def __init__(self, block, edges=None):
        self.block = block
        self.edges = []

class DAG(object):

    def __init__(self):
        """ Construct a new DAG with no nodes or edges. """
        self.graph = {}

    def add_node(self, block):
        """ Add a node if it does not exist yet, or error out. """
        if block.node_id in self.graph:
            raise KeyError('node %s already exists' % block.node_id)
        self.graph[block.node_id] = DAGNode(block)

    def add_node_chain(self, node_id, graph):
        self.graph[node_id] = graph[node_id]
        for blocks in self.predecessors(node_id, graph):
            self.add_node_chain(blocks, graph)

    def leaf_selection(self, block, gnode_id, X_CENTER, Y_CENTER):
        x1 = block.coords[1]
        y1 = block.coords[2]

        def compare_x(a):
            if x1 < X_CENTER:
                return a < X_CENTER
            else:
                return a > X_CENTER

        def compare_y(a):
            if y1 < Y_CENTER:
                return a < Y_CENTER
            else:
                return a > Y_CENTER

        for node_id in self.ind_nodes():
            x2 = self.graph[node_id].block.coords[1]
            y2 = self.graph[node_id].block.coords[2]
            if node_id != block.node_id and compare_x(x2) and compare_y(y2):
                return node_id
        return gnode_id

    def leaf_selection2(self, block):
        return self.ind_nodes()[0]

    def merkle_hash(self, block):
        hashlist = []
        for predecessor in self.all_downstreams2(block.node_id):
            hashlist.append(hash(self.graph[predecessor].block))
        return BU.hash(tuple(hashlist))

    def add_node_if_not_exists(self, block, graph=None):
        try:
            self.add_node(block)
        except KeyError:
            pass

    def delete_node(self, block, graph=None):
        """ Deletes this node and all edges referencing it. """
        if not graph:
            graph = self.graph
        if block.node_id not in graph:
            raise KeyError('node %s does not exist' % block.node_id)
        graph.pop(block)

        for node, edges in six.iteritems(graph):
            if block.node_id in edges:
                edges.remove(block)

    def delete_node_if_exists(self, block, graph=None):
        try:
            self.delete_node(block, graph=graph)
        except KeyError:
            pass

    def add_edge(self, id1, id2, graph=None):
        """ Add an edge (dependency) between the specified nodes. """
        if not graph:
            graph = self.graph
        if id1 not in graph or id2 not in graph:
            raise KeyError('one or more nodes do not exist in graph')
        test_graph = deepcopy(graph)
        test_graph[id1].edges.append(id2)
        graph[id1].edges.append(id2)
        # is_valid, message = self.validate(test_graph)
        # if is_valid:
        #     graph[id1].edges.append(id2)
        # else:
        #     raise DAGValidationError(message)

    def delete_edge(self, ind_node, dep_node, graph=None):
        """ Delete an edge from the graph. """
        if not graph:
            graph = self.graph
        if dep_node not in graph.get(ind_node, []):
            raise KeyError('this edge does not exist in graph')
        graph[ind_node].remove(dep_node)

    def rename_edges(self, old_task_name, new_task_name, graph=None):
        """ Change references to a task in existing edges. """
        if not graph:
            graph = self.graph
        for node, edges in graph.items():

            if node == old_task_name:
                graph[new_task_name] = copy(edges)
                del graph[old_task_name]

            else:
                if old_task_name in edges:
                    edges.remove(old_task_name)
                    edges.add(new_task_name)

    def predecessors(self, node_id, graph=None):
        """ Returns a list of all predecessors of the given node """
        if graph is None:
            graph = self.graph
        return [key for key in graph if node_id in graph[key].edges]
    
    def all_predecessors(self, node_id, graph=None):
        allpredecessors = []
        if graph is None:
            graph = self.graph
        for nodes in self.predecessors(node_id):
            allpredecessors += [nodes]
            allpredecessors += self.predecessors(nodes, graph)
        return allpredecessors

    def downstream(self, node_id, graph=None):
        """ Returns a list of all nodes this node has edges towards. """
        if graph is None:
            graph = self.graph
        if node_id not in graph:
            raise KeyError('node %s is not in graph' % node_id)
        return list(graph[node_id].edges)

    def all_downstreams2(self, node_id, graph=None):
        alldownstreams = []
        if graph is None:
            graph = self.graph
        for nodes in self.downstream(node_id):
            alldownstreams += [nodes]
            alldownstreams += self.all_downstreams2(nodes, graph)
        return alldownstreams

    def all_downstreams(self, node_id, graph=None):
        """Returns a list of all nodes ultimately downstream
        of the given node in the dependency graph, in
        topological order."""
        if graph is None:
            graph = self.graph
        nodes = [graph[node_id].edges]
        nodes_seen = set()
        i = 0
        while i < len(nodes):
            downstreams = self.downstream(nodes[i], graph)
            for downstream_node in downstreams:
                if downstream_node not in nodes_seen:
                    nodes_seen.add(downstream_node)
                    nodes.append(downstream_node)
            i += 1
        return list(
            filter(
                lambda node: node in nodes_seen,
                self.topological_sort(graph=graph)
            )
        )

    def all_leaves(self, graph=None):
        """ Return a list of all leaves (nodes with no downstreams) """
        if graph is None:
            graph = self.graph
        return [key for key in graph if not graph[key].edges]

    def from_dict(self, graph_dict):
        """ Reset the graph and build it from the passed dictionary.

        The dictionary takes the form of {node_name: [directed edges]}
        """

        self.reset_graph()
        for new_node in six.iterkeys(graph_dict):
            self.add_node(new_node)
        for ind_node, dep_nodes in six.iteritems(graph_dict):
            if not isinstance(dep_nodes, list):
                raise TypeError('dict values must be lists')
            for dep_node in dep_nodes:
                self.add_edge(ind_node, dep_node)

    def ind_nodes(self, graph=None):
        """ Returns a list of all nodes in the graph with no dependencies. """
        if graph is None:
            graph = self.graph

        dependent_nodes = set(
            node for dependents in six.itervalues(graph) for node in dependents.edges
        )
        return [node for node in graph.keys() if node not in dependent_nodes]

    def validate(self, graph=None):
        """ Returns (Boolean, message) of whether DAG is valid. """
        graph = graph if graph is not None else self.graph
        if len(self.ind_nodes(graph)) == 0:
            return (False, 'no independent nodes detected')
        try:
            self.topological_sort(graph)
        except ValueError:
            return (False, 'failed topological sort')
        return (True, 'valid')

    def topological_sort(self, graph=None):
        """ Returns a topological ordering of the DAG.

        Raises an error if this is not possible (graph is not valid).
        """
        if graph is None:
            graph = self.graph

        in_degree = {}
        for u in graph:
            in_degree[u] = 0

        for u in graph:
            for v in graph[u].edges:
                in_degree[v] += 1

        queue = deque()
        for u in in_degree:
            if in_degree[u] == 0:
                queue.appendleft(u)

        l = []
        while queue:
            u = queue.pop()
            l.append(u)
            for v in graph[u].edges:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.appendleft(v)

        if len(l) == len(graph):
            return l
        else:
            raise ValueError('graph is not acyclic')

    def size(self):
        return len(self.graph)