import sys
import time
import json
import flask
import warnings

import networkx as nx

from copy import deepcopy
from typing import Optional, Union, Dict, Tuple, Generator

from utls import NpEncoder
from regret_synthesis_toolbox.src.graph import TwoPlayerGraph

app = flask.Flask(__name__, static_folder="force")

@app.route("/")
def static_proxy():
    return app.send_static_file("tree_dfs_all_attrs_scrollable.html")


class InteractiveGraph():
    """
     A Class that contains all the necesary tools to plot a interactive tree using D3.js package
    """
    
    @staticmethod
    def visualize_game(game, depth_limit: None):
        """
         Main method to visualize the gam. We first run a DFS on the game graph and then construct a tree for the given depth limit. 
         Then, we dump the tree data to a json file and serve it over http using D3.js package.
        """
        # call NEtworkX and construct Tree for a given depth limit
        if depth_limit is None:
            depth_limit = len(game._graph)

        # run bfs/dfs upto certian depth
        dfs_tree = TreeTraversalMyGame(game=game)
        dfs_tree.tree_traversal(bfs=True, dfs=False, depth_limit=depth_limit)
        d = TreeTraversalMyGame.tree_data(G=dfs_tree._tree, root=game.get_initial_states()[0][0], ident="parent")
        
        # write json file
        json.dump(d, open("force/tree_dfs.json", "w"), cls=NpEncoder)
        print("Wrote node-link JSON data to force/force.json")
        
        # Serve the file over http to allow for cross origin requests
        print("\nGo to http://localhost:8000 to see the graph\n")
        app.run(port=8000)


class TreeTraversalMyGame():
    
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
                d = {"name": str(child), "edge_name": G[n][child]['actions'], "label": G.nodes[child].get('ap'), "player": G.nodes[child]['player'], ident: child}
                c = add_children(child, G)
                if c:
                    d[children] = c
                children_.append(d)
            return children_

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
                    self._tree.add_edge(u_node, v_node, **self.game._graph[u_node][v_node][0])

    def construct_tree_dfs(self, depth_limit: Union[int, None]) -> Generator[Tuple, Tuple, Dict]:
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
    

    def construct_tree_bfs(self, depth_limit: Union[int, None]) -> Generator[Tuple, Tuple, Dict]:
        """
        This method constructs a tree in a non-recurisve breadth first fashion for all plays in the original graph whose depth < depth_limit.
        """
        source = self.game.get_initial_states()[0][0]

        if depth_limit is None:
            depth_limit = len(self.game)

        visited = set()
        visited.add(source)
        next_parents_children = [(source, iter(self.game._graph[source]))] 
        depth_now = 0
        while next_parents_children and depth_now < depth_limit:
            this_parents_children = next_parents_children
            next_parents_children = []
            for parent, children in this_parents_children:                
                for child in children:
                    if child not in visited:
                        yield ((parent), self.game._graph.nodes[parent]), ((child), self.game._graph.nodes[child]), {'weight': 0}
                        visited.add(child)
                        next_parents_children.append((child, iter(self.game._graph[child])))
                if len(visited) == len(self.game._graph):
                    return
            depth_now += 1

        print(f"Done with BFS: {len(visited)}") 


    def tree_traversal(self, bfs: bool, dfs: bool = False, depth_limit: Optional[int] = None):
        """
         Parent method to call the traversal.
        """

        start = time.time()
        # if bfs and dfs is False:
        #     warnings.warn("[Error] Please set either bfs or dfs to true")
        #     sys.exit(-1)
        
        # if bfs and dfa

        self._tree.add_node(self.game.get_initial_states()[0][0])
        if bfs:
            self.add_edges(self.construct_tree_bfs(depth_limit))
        elif dfs:
            self.add_edges(self.construct_tree_dfs(depth_limit))
        else:
            warnings.warn("[Error] Please set either bfs or dfs to true")
            sys.exit(-1)
        
        stop = time.time()
        print(f"Time to construct the Tree is: {stop - start:.2f}")
       