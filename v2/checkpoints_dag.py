


# also, from a minimal dag-like blockchain implementation








# Checkpoint: Agent metadata, Cog. Arch. (with Short-Term Memory storage), Optimizer State(?), etc





"""
Hedera Hashgraph Implementation in Python

A distributed consensus algorithm based on the Hashgraph data structure.
Implements the core principles of asynchronous Byzantine Fault Tolerant (aBFT) consensus.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from enum import Enum
import heapq


class Signature:
    """Represents a digital signature using SHA-256"""
    def __init__(self, data: str):
        self.data = data
        self.hash = hashlib.sha256(data.encode()).hexdigest()
    
    def __eq__(self, other):
        return self.hash == other.hash
    
    def __repr__(self):
        return f"Signature({self.hash[:16]}...)"


@dataclass
class Event:
    """
    Represents a node/event in the hashgraph.
    
    Each event contains:
    - creator: ID of the node that created this event
    - timestamp: when the event was created
    - parents: references to parent events (one from same creator, one other)
    - transactions: payload data
    - signature: creator's signature
    - round: consensus round number
    - is_famous: whether this event is famous (used in consensus)
    """
    creator: int
    timestamp: float
    parents: Tuple[Optional['Event'], Optional['Event']]  # (self_parent, other_parent)
    transactions: List[str] = field(default_factory=list)
    signature: Optional[Signature] = None
    round: Optional[int] = None
    is_famous: Optional[bool] = None
    generation: int = 0
    
    def __post_init__(self):
        if self.signature is None:
            # Create signature from event data
            event_data = f"{self.creator}{self.timestamp}{self.transactions}"
            self.signature = Signature(event_data)
        
        # Calculate generation (distance from root)
        if self.parents[0] is not None or self.parents[1] is not None:
            parent_gens = [p.generation for p in self.parents if p is not None]
            self.generation = max(parent_gens) + 1 if parent_gens else 1
    
    def get_hash(self) -> str:
        """Get unique hash of this event"""
        return self.signature.hash
    
    def __repr__(self):
        return f"Event(creator={self.creator}, gen={self.generation}, hash={self.get_hash()[:8]}...)"
    
    def __lt__(self, other):
        """For heap operations"""
        return self.generation < other.generation


@dataclass
class Witness:
    """A witness is an event that is famous and can participate in consensus"""
    event: Event
    round: int
    is_famous: bool = True


class RoundCreationRule(Enum):
    """Determines when a new round begins"""
    SUPERMAJORITY = "supermajority"  # When supermajority sees previous round


class VirtualVoting:
    """
    Implements virtual voting mechanism for determining famous witnesses
    """
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.supermajority_threshold = (2 * num_nodes) // 3 + 1
    
    def can_strongly_see(self, event1: Event, event2: Event) -> bool:
        """
        Event1 can strongly see Event2 if Event1 can see Event2,
        and more than 2/3 of the witnesses that Event1 can see
        can also see Event2.
        """
        # Simplified: assume direct ancestor relationships form a DAG
        # In full implementation, track actual ancestor relationships
        return event1.generation > event2.generation
    
    def is_witness(self, event: Event) -> bool:
        """
        A witness is an event that is the first event by a node
        in a particular round.
        """
        return True  # Simplified: all events are potential witnesses
    
    def decide_fame(self, witness1: Event, witness2: Event) -> Optional[bool]:
        """
        Use virtual voting to determine if witness2 is famous
        from the perspective of witness1.
        Returns True if famous, False if not famous, None if undecided.
        """
        # Simplified voting mechanism
        # In full implementation: recursive voting based on ancestor relationships
        if witness1.generation >= witness2.generation:
            return True
        return False


class Hashgraph:
    """
    Main Hashgraph implementation.
    Manages the DAG of events and performs consensus.
    """
    
    def __init__(self, num_nodes: int, node_id: int):
        """
        Initialize a Hashgraph instance for a specific node.
        
        Args:
            num_nodes: Total number of nodes in the network
            node_id: ID of this node (0 to num_nodes-1)
        """
        self.num_nodes = num_nodes
        self.node_id = node_id
        self.supermajority_threshold = (2 * num_nodes) // 3 + 1
        
        # Data structures
        self.events: Dict[str, Event] = {}  # hash -> event
        self.event_by_creator: Dict[int, List[Event]] = defaultdict(list)
        self.witnesses_by_round: Dict[int, List[Event]] = defaultdict(list)
        self.famous_witnesses_by_round: Dict[int, List[Event]] = defaultdict(list)
        self.rounds: Dict[Event, int] = {}  # event -> round number
        self.virtual_voting = VirtualVoting(num_nodes)
        
        # Consensus state
        self.current_round = 0
        self.consensus_events: List[Event] = []
        self.event_order: List[Event] = []
        
        # Last event by each creator
        self.last_event: Dict[int, Optional[Event]] = {i: None for i in range(num_nodes)}
        
        # Round assignments (hash -> round)
        self.event_rounds: Dict[str, int] = {}
    
    def create_event(self, transactions: List[str]) -> Event:
        """
        Create a new event for this node.
        
        Args:
            transactions: List of transaction hashes
        
        Returns:
            The created event
        """
        # Get parent events
        self_parent = self.last_event[self.node_id]
        
        # Get other parent from node with most recent event we can see
        other_parent = None
        if self.last_event:
            max_gen = -1
            for creator_id in range(self.num_nodes):
                if creator_id != self.node_id:
                    event = self.last_event[creator_id]
                    if event and event.generation > max_gen:
                        max_gen = event.generation
                        other_parent = event
        
        # Create event
        event = Event(
            creator=self.node_id,
            timestamp=time.time(),
            parents=(self_parent, other_parent),
            transactions=transactions
        )
        
        # Add to graph
        self.events[event.get_hash()] = event
        self.event_by_creator[self.node_id].append(event)
        self.last_event[self.node_id] = event
        
        return event
    
    def receive_event(self, event: Event) -> None:
        """
        Receive an event from another node.
        
        Args:
            event: The event to add to the hashgraph
        """
        if event.get_hash() not in self.events:
            self.events[event.get_hash()] = event
            self.event_by_creator[event.creator].append(event)
            self.last_event[event.creator] = event
    
    def decide_fame(self) -> None:
        """
        Run the fame decision algorithm to identify famous witnesses.
        Uses virtual voting to reach consensus.
        """
        # Get all unique round numbers
        all_events = list(self.events.values())
        
        if not all_events:
            return
        
        # Assign rounds if not already assigned
        self._assign_rounds()
        
        # Collect witnesses by round
        for event in all_events:
            if self._is_witness(event):
                round_num = self.rounds.get(event, 0)
                self.witnesses_by_round[round_num].append(event)
        
        # Determine fame for witnesses
        for round_num in sorted(self.witnesses_by_round.keys()):
            witnesses = self.witnesses_by_round[round_num]
            for witness in witnesses:
                is_famous = self._determine_fame(witness, round_num)
                witness.is_famous = is_famous
                
                if is_famous:
                    self.famous_witnesses_by_round[round_num].append(witness)
    
    def _is_witness(self, event: Event) -> bool:
        """Check if an event is a witness (first event in a round from its creator)"""
        round_num = self.rounds.get(event, 0)
        events_by_creator = defaultdict(list)
        
        for e in self.events.values():
            if self.rounds.get(e, 0) == round_num:
                events_by_creator[e.creator].append(e)
        
        if event.creator in events_by_creator:
            creator_events = sorted(events_by_creator[event.creator], 
                                  key=lambda e: e.generation)
            return creator_events[0] == event if creator_events else False
        return False
    
    def _assign_rounds(self) -> None:
        """Assign round numbers to events using topological ordering"""
        if not self.events:
            return
        
        # Find root events (no parents)
        assigned = set()
        current_round = 0
        
        # Use BFS-like approach
        queue = []
        for event in self.events.values():
            if self.rounds.get(event) is None:
                if event.parents[0] is None and event.parents[1] is None:
                    self.rounds[event] = 0
                    assigned.add(event.get_hash())
                    queue.append(event)
        
        # Assign remaining events
        while queue:
            current_event = queue.pop(0)
            current_round = self.rounds.get(current_event, 0)
            
            # Find children of current event
            for event in self.events.values():
                if event.get_hash() not in assigned:
                    parent_hashes = {p.get_hash() for p in event.parents if p}
                    if current_event.get_hash() in parent_hashes:
                        parent_rounds = [self.rounds.get(p, 0) for p in event.parents if p]
                        self.rounds[event] = max(parent_rounds) + 1 if parent_rounds else current_round + 1
                        assigned.add(event.get_hash())
                        queue.append(event)
    
    def _determine_fame(self, witness: Event, round_num: int) -> bool:
        """
        Determine if a witness is famous using simplified voting.
        In full implementation, this uses recursive virtual voting.
        """
        # Count how many witnesses in next round can see this witness
        next_round = round_num + 1
        if next_round not in self.witnesses_by_round:
            return False
        
        next_round_witnesses = self.witnesses_by_round[next_round]
        
        # Check if witness can be seen by supermajority in next round
        can_see_count = sum(
            1 for w in next_round_witnesses 
            if self.virtual_voting.can_strongly_see(w, witness)
        )
        
        return can_see_count >= self.supermajority_threshold
    
    def find_consensus_order(self) -> List[Event]:
        """
        Determine the consensus order of events.
        Famous witnesses in each round determine the order.
        
        Returns:
            List of events in consensus order
        """
        consensus = []
        
        # Order by famous witnesses' rounds
        for round_num in sorted(self.famous_witnesses_by_round.keys()):
            famous = self.famous_witnesses_by_round[round_num]
            
            # Sort witnesses by timestamp
            sorted_famous = sorted(famous, key=lambda e: e.timestamp)
            
            # Add events referenced by these witnesses (via transitive closure)
            for witness in sorted_famous:
                consensus.append(witness)
        
        self.event_order = consensus
        return consensus
    
    def get_statistics(self) -> Dict:
        """Get statistics about the hashgraph"""
        return {
            'total_events': len(self.events),
            'events_by_creator': dict(self.event_by_creator),
            'current_round': self.current_round,
            'famous_witnesses': sum(len(w) for w in self.famous_witnesses_by_round.values()),
            'consensus_order_length': len(self.event_order)
        }


class HashgraphNetwork:
    """
    Simulates a network of Hashgraph nodes for testing and demonstration.
    """
    
    def __init__(self, num_nodes: int):
        """Initialize a network of hashgraph nodes"""
        self.num_nodes = num_nodes
        self.nodes: Dict[int, Hashgraph] = {
            i: Hashgraph(num_nodes, i) for i in range(num_nodes)
        }
        self.message_log: List[Tuple[int, int, Event]] = []  # (from, to, event)
    
    def node_create_event(self, node_id: int, transactions: List[str]) -> Event:
        """
        Have a node create and broadcast a new event.
        
        Args:
            node_id: ID of the node creating the event
            transactions: List of transaction hashes
        """
        event = self.nodes[node_id].create_event(transactions)
        
        # Broadcast to all other nodes
        for other_id in range(self.num_nodes):
            if other_id != node_id:
                self.nodes[other_id].receive_event(event)
                self.message_log.append((node_id, other_id, event))
        
        return event
    
    def run_consensus(self) -> Dict:
        """
        Run the consensus algorithm across the network.
        
        Returns:
            Consensus results from all nodes
        """
        results = {}
        
        for node_id, node in self.nodes.items():
            node.decide_fame()
            consensus_order = node.find_consensus_order()
            results[node_id] = {
                'consensus_order': [e.get_hash()[:8] for e in consensus_order],
                'stats': node.get_statistics()
            }
        
        return results
    
    def simulate_transactions(self, num_rounds: int, txns_per_round: int = 3) -> None:
        """
        Simulate nodes creating transactions over multiple rounds.
        
        Args:
            num_rounds: Number of rounds to simulate
            txns_per_round: Number of transactions per round
        """
        for round_num in range(num_rounds):
            for node_id in range(self.num_nodes):
                transactions = [
                    f"txn_{round_num}_{node_id}_{i}" 
                    for i in range(txns_per_round)
                ]
                self.node_create_event(node_id, transactions)
    
    def get_network_stats(self) -> Dict:
        """Get statistics about the entire network"""
        return {
            'num_nodes': self.num_nodes,
            'total_messages': len(self.message_log),
            'node_stats': {
                node_id: node.get_statistics() 
                for node_id, node in self.nodes.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Hedera Hashgraph Implementation Demo")
    print("=" * 70)
    
    # Create a 4-node network
    network = HashgraphNetwork(num_nodes=4)
    
    print("\n1. Simulating distributed transactions...")
    network.simulate_transactions(num_rounds=2, txns_per_round=2)
    
    print("2. Running consensus algorithm...")
    results = network.run_consensus()
    
    print("\n3. Consensus Results:")
    for node_id, result in results.items():
        print(f"\n   Node {node_id}:")
        print(f"   - Consensus order length: {len(result['consensus_order'])}")
        print(f"   - Total events: {result['stats']['total_events']}")
        print(f"   - Famous witnesses: {result['stats']['famous_witnesses']}")
    
    print("\n4. Network Statistics:")
    stats = network.get_network_stats()
    print(f"   - Total nodes: {stats['num_nodes']}")
    print(f"   - Total messages: {stats['total_messages']}")
    
    print("\n5. Event DAG Structure:")
    node_0 = network.nodes[0]
    print(f"   - Total events in graph: {len(node_0.events)}")
    print(f"   - Events by creator:")
    for creator_id, events in sorted(node_0.event_by_creator.items()):
        print(f"     - Node {creator_id}: {len(events)} events")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)