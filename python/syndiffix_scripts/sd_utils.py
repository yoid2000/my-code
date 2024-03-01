from syndiffix import Synthesizer
from syndiffix.tree import Node, Leaf, Branch

class TreeDump:
    def __init__(self, path):
        self.path = path

    def dump_trees(synth: Synthesizer):
        def dump_tree_walk(node: Node) -> None:
            if isinstance(node, Leaf):
                low_threshold = node.context.anonymization_context.anonymization_params.low_count_params.low_threshold
                if node.is_singularity() and node.is_over_threshold(low_threshold):
                    print(f"Leaf: {node.snapped_intervals}")
            elif isinstance(node, Branch):
                for child_node in node.children.values():
                    dump_tree_walk(child_node)

        for col_id, root in synth.forest._tree_cache.items():
            print(col_id)
            dump_tree_walk(root)
        quit()