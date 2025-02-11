
import time
from copy import deepcopy
import networkx as nx

from typing import Optional, Union, List, Iterable, Dict, Set, Tuple, Generator

from regret_synthesis_toolbox.src.graph import TwoPlayerGraph


class DFSPersonal():
    
    def __init__(self, game):
        self._tree =  nx.DiGraph()
        self.game: TwoPlayerGraph = deepcopy(game)
    
    @staticmethod
    def tree_data(G, root, ident="id", children="children"):
        """Returns data in tree format that is suitable for JSON serialization
        and use in JavaScript documents.

        Parameters
        ----------
        G : NetworkX graph
        G must be an oriented tree

        root : node
        The root of the tree

        ident : string
            Attribute name for storing NetworkX-internal graph data. `ident` must
            have a different value than `children`. The default is 'id'.

        children : string
            Attribute name for storing NetworkX-internal graph data. `children`
            must have a different value than `ident`. The default is 'children'.

        Returns
        -------
        data : dict
        A dictionary with node-link formatted data.

        Raises
        ------
        NetworkXError
            If `children` and `ident` attributes are identical.

        Examples
        --------
        >>> from networkx.readwrite import json_graph
        >>> G = nx.DiGraph([(1, 2)])
        >>> data = json_graph.tree_data(G, root=1)

        To serialize with json

        >>> import json
        >>> s = json.dumps(data)

        Notes
        -----
        Node attributes are stored in this format but keys
        for attributes must be strings if you want to serialize with JSON.

        Graph and edge attributes are not stored.

        See Also
        --------
        tree_graph, node_link_data, adjacency_data
        """
        if ident == children:
            raise nx.NetworkXError("The values for `id` and `children` must be different.")

        def add_children(n, G):
            nbrs = G[n]
            if len(nbrs) == 0:
                return []
            children_ = []
            for child in nbrs:
                # d = {**G.nodes[child], ident: child}
                d = {"name": str(child), "edge_name": G[n][child]['actions'], "label": G.nodes[child].get('ap'), "player": G.nodes[child]['player'], ident: child}
                c = add_children(child, G)
                if c:
                    d[children] = c
                children_.append(d)
            return children_

        # return {**G.nodes[root], ident: root, children: add_children(root, G)}
        return {"name": str(root), "player": G.nodes[root]['player'], "label": G.nodes[root].get('ap'), ident: root, children: add_children(root, G)}

    def add_edges(self, ebunch_to_add, **attr) -> None:
            """
            A function to add all the edges in the ebunch_to_add. 
            """
            init_node_added: bool = False
            for e in ebunch_to_add:
                # u and v as 2-Tuple with node and node attributes of source (u) and destination node (v)
                u, v, dd = e
                u_node = u[0]
                u_attr = u[1]
                v_node = v[0]
                v_attr = v[1]
                ddd = {}
                ddd.update(attr)
                ddd.update(dd)

                # add node attributes too
                self._tree.add_node(u_node, **u_attr)
                self._tree.add_node(v_node, **v_attr)

                if u_attr.get('init') and not init_node_added:
                    init_node_added = True
                
                if init_node_added and 'init' in u_attr:
                    del u_attr['init']
                
                # add edge attributes too 
                if not self._tree.has_edge(u_node, v_node):
                    # self.game._graph
                    self._tree.add_edge(u_node,v_node, **self.game._graph[u_node][v_node][0])
                    # self._tree[u_node][v_node][key].update(ddd)

    def construct_tree(self, depth_limit: int) -> Generator[Tuple, Tuple, Dict]:
        """
        This method constructs a tree in a non-recurisve depth first fashion for all plays in the original graph whose depth < depth_limit.
        """
        source = self.game.get_initial_states()[0][0]
        nodes = [source]

        if depth_limit is None:
            depth_limit = len(self.game)

        visited = set()
        for start in nodes:
            if start in visited:
                continue

            visited.add(start)
            stack = [(start, iter(self.game._graph[start]))] 
            depth_now = 1
            while stack:
                parent, children = stack[-1]
                for child in children:                
                    if child not in visited:
                        yield ((parent), self.game._graph.nodes[parent]), ((child), self.game._graph.nodes[child]), {'weight': 0}

                    visited.add(child)

                    if depth_now < depth_limit:
                        stack.append((child, iter(self.game._graph[child])))
                        depth_now += 1
                        break
                else:
                    stack.pop()
                    depth_now -= 1 


    def tree_dfs(self, depth_limit: None):
        start = time.time()
        
        self._tree.add_node(self.game.get_initial_states()[0][0])
        # terminal_state: str = "vT"
        # self._adm_tree.add_state(terminal_state, **{'init': False, 'accepting': False})

        self.add_edges(self.construct_tree(depth_limit))
        stop = time.time()
        print(f"Time to construct the Tree is: {stop - start:.2f}")
       