from enum import Enum, auto


class OpType(Enum):
    Load = (auto(), "load")
    Add = (auto(), "+")
    Mul = (auto(), "*")
    Pow = (auto(), "**")
    Neg = (auto(), "-1*")
    Div = (auto(), "/")

    def __init__(self, id, label):
        self._id = id
        self._label = label

    def __str__(self):
        return self._label
    
    def __repr__(self):
        return self._label
    

class Scalar:
    def __init__(self, data, prev=(), op=OpType.Load):
        self.data = data
        self.prev = prev
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), OpType.Add)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), OpType.Mul)
        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Scalar(self.data**other, (self, other), OpType.Pow)
        return out

    def __truediv__(self, other):  # self / other
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data / other.data, (self, other), OpType.Div)
        return out

    def __neg__(self):
        out = Scalar(-1 * self.data, (self,), OpType.Neg)
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __rtruediv__(self, other):  # other / self
        return other / self

    def __repr__(self):
        return f"Scalar(data={self.data})"


def trace(root) -> tuple[list[Scalar], list[tuple[Scalar, Scalar]]]:
    # traces the full graph of nodes and edges starting from the root
    nodes, edges = [], []

    def build(v):
        if v not in nodes:
            nodes.append(v)
            for parent in v.prev:
                if (parent, v) not in edges:
                    edges.append((parent, v))
                build(parent)

    build(root)
    return nodes, edges

def draw_comp_graph(root: Scalar, format="svg", rankdir="LR", outfile="graph"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    from graphviz import Digraph

    assert rankdir in ["LR", "TB"]
    dot = Digraph(
        format=format,
        graph_attr={"rankdir": rankdir, "nodesep": "0.1", "ranksep": "0.4"},
    )

    nodes, edges = trace(root)
    for n in nodes:
        # fillcolor = n._vis_color if hasattr(n, "_vis_color") else "white"
        dot.node(
            name=str(id(n)),
            label=f"data: {n.data:.4f}\n",
            shape="box",
            style="filled",
            # fillcolor=fillcolor,
            width="0.1",
            height="0.1",
            fontsize="10",
        )
        if n.op:
            dot.node(
                name=str(id(n)) + str(n.op),
                label=str(n.op),
                width="0.1",
                height="0.1",
                fontsize="10",
            )
            dot.edge(str(id(n)) + str(n.op), str(id(n)), minlen="1")

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + str(n2.op), minlen="1")

    print("found a total of ", len(nodes), "nodes and", len(edges), "edges")
    print("saving graph to", outfile + "." + format)
    dot.render(outfile, format=format)


def count_ops(root: Scalar) -> dict[str, int]:
    nodes, _ = trace(root)
    ops = {}
    for n in nodes:
        if n.prev:
            ops[n.prev] = ops.get(n.prev, 0) + 1

    return ops
