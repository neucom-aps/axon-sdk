# Copyright (C) 2025  Neucom Aps
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from enum import Enum, auto

from typing import Self


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

    def __add__(self, other) -> 'Scalar':  # self + other
        assert can_proceed(other), f'Wrong datatype for {other}'
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), OpType.Add)
        return out

    def __mul__(self, other) -> 'Scalar':  # self * other
        assert can_proceed(other), f'Wrong datatype for {other}'
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), OpType.Mul)
        return out

    def __pow__(self) -> 'Scalar':
        raise Exception('Op not supported yet')

    def __truediv__(self, other) -> 'Scalar':  # self / other
        assert can_proceed(other), f'Wrong datatype for {other}'
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data / other.data, (self, other), OpType.Div)
        return out

    def __neg__(self) -> 'Scalar':  # -self
        out = Scalar(-1 * self.data, (self,), OpType.Neg)
        return out

    def __radd__(self, other) -> 'Scalar':  # other + self
        assert can_proceed(other), f'Wrong datatype for {other}'
        return self + other

    def __sub__(self, other) -> 'Scalar':  # self - other
        assert can_proceed(other), f'Wrong datatype for {other}'
        return self + (-other)

    def __rsub__(self, other) -> 'Scalar':  # other - self
        assert can_proceed(other), f'Wrong datatype for {other}'
        return other + (-self)

    def __rmul__(self, other) -> 'Scalar':  # other * self
        assert can_proceed(other), f'Wrong datatype for {other}'
        return self * other

    def __rtruediv__(self, other) -> 'Scalar':  # other / self
        assert can_proceed(other), f'Wrong datatype for {other}'
        return other / self

    def __repr__(self) -> str:
        return f"Scalar(data={self.data})"

    def draw_comp_graph(self, outfile="graph", format="svg", rankdir="LR"):
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

        nodes, edges = trace(self)
        for n in nodes:
            dot.node(
                name=str(id(n)),
                label=f"data: {n.data:.4f}\n",
                shape="box",
                style="filled",
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

def can_proceed(value) -> bool:
    if isinstance(value, int):
        return True
    elif isinstance(value, float):
        return True
    elif isinstance(value, Scalar):
        return True
    else:
        return False