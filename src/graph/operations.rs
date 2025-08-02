use super::{Graph, NodeId, EdgeId};

pub trait GraphOps {
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn has_node(&self, id: NodeId) -> bool;
    fn has_edge(&self, id: EdgeId) -> bool;
    fn neighbors(&self, node_id: NodeId) -> Vec<NodeId>;
}

impl GraphOps for Graph {
    fn node_count(&self) -> usize {
        self.get_nodes().count()
    }

    fn edge_count(&self) -> usize {
        self.get_edges().count()
    }

    fn has_node(&self, id: NodeId) -> bool {
        self.get_node(id).is_ok()
    }

    fn has_edge(&self, id: EdgeId) -> bool {
        self.get_edge(id).is_ok()
    }

    fn neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.get_edges_from(node_id)
            .map(|e| e.target)
            .collect()
    }
}