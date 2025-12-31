#  tree structures, mostly for taxonomic neighbor finding. this is mostly a place holder until i figure out if this functionality is trulyt needed, needed from scratch, or can be sourced from phylotree-rs
import heapq
from collections import defaultdict, deque

import polars as pl


class TaxonomyTree:
    """
    taxonomy tree for finding nearest neighbors with available data.

    This implementation uses:
    - Dictionary-based adjacency lists for parent/child lookups # this might be memory heavy, mihgt need to rethink.
    - BFS for finding nearest ancestors/descendants with data
    - Caching for repeated queries (not sure this is actually useful)
    - Priority-based selection when multiple neighbors at same distance

    # TODO: look at:
    # bigtree - seems to have built-in "from polars" fucntions
    # ete3 - might have some fucntions and utils for NCBI taxonomy
    # networkx - might be efficent
    # tskit - might be efficent + modern (?)
    # phylotree - rust backend, not a lot of features but seems cool. submitted merge request to make it compatible with python<3.11

    """

    def __init__(
        self, nodes_df, data_availability_df=None, priority_columns=None
    ):
        """
        Initialize taxonomy tree.

        Args:
            nodes_df: DataFrame with tax_id, parent_tax_id, rank, scientific_name
            data_availability_df: DataFrame with tax_id and availability flags/metadata
                                (e.g., has_genome, protein_coding_gene_count, genome_size)
            priority_columns: List of column names to use for prioritization when selecting
                            among neighbors at same distance. Higher values = higher priority.
                            Example: ['protein_coding_gene_count', 'genome_size']
        """
        self.nodes_df = nodes_df
        self.parent_map = {}  # tax_id -> parent_tax_id
        self.children_map = defaultdict(
            list
        )  # parent_tax_id -> [child_tax_ids]
        self.rank_map = {}  # tax_id -> rank
        self.name_map = {}  # tax_id -> scientific_name
        self.data_available = set()  # tax_ids with available data
        self.metadata_map = {}  # tax_id -> dict of metadata (gene counts, genome size, etc.)
        self.cache = {}  # Cache for repeated queries
        self.priority_columns = priority_columns or []
        self.leaf_nodes = set()  # Track leaf nodes

        self._build_tree()
        if data_availability_df is not None:
            self._set_data_availability(data_availability_df)

    def _build_tree(self):
        """Build internal tree structure from nodes DataFrame"""
        for row in self.nodes_df.iter_rows(named=True):
            tax_id = row["tax_id"]
            parent_tax_id = row["parent_tax_id"]

            self.parent_map[tax_id] = parent_tax_id
            self.children_map[parent_tax_id].append(tax_id)

            # Store additional info if available
            if "rank" in row:
                self.rank_map[tax_id] = row["rank"]
            if "scientific_name" in row:
                self.name_map[tax_id] = row["scientific_name"]

        # Identify leaf nodes (nodes with no children)
        self.leaf_nodes = {
            tax_id
            for tax_id in self.parent_map.keys()
            if tax_id not in self.children_map
            or len(self.children_map[tax_id]) == 0
        }
        print(
            f"Built tree with {len(self.parent_map)} nodes, {len(self.leaf_nodes)} leaf nodes"
        )

    def _set_data_availability(self, data_df):
        """Set which tax_ids have available data and store metadata"""
        # Store all metadata for each tax_id
        available_tax_ids = []

        for row in data_df.iter_rows(named=True):
            tax_id = row["tax_id"]

            # Store all metadata for this tax_id (excluding tax_id itself)
            metadata = {k: v for k, v in row.items() if k != "tax_id"}
            self.metadata_map[tax_id] = metadata

            # Check if any data columns indicate availability
            has_data = any(
                row.get(col) and row.get(col) not in ["", "na", "NA", "-", None]
                for col in data_df.columns
                if col != "tax_id"
            )
            if has_data:
                available_tax_ids.append(tax_id)

        self.data_available = set(available_tax_ids)
        print(f"Set data availability for {len(self.data_available)} tax_ids")
        print(f"Stored metadata for {len(self.metadata_map)} tax_ids")

    def propagate_data_to_ancestors(self, aggregation_method="count"):
        """
        Propagate data availability from leaf nodes up to their ancestors.
        This helps fill in data for intermediate nodes based on their descendants.

        Args:
            aggregation_method: How to aggregate child data to parents
                'count' - count of descendants with data
                'any' - mark parent as having data if any child has data
                'majority' - mark parent as having data if >50% of children have data
                'summary' - create summary metadata from all descendants

        Returns:
            Number of nodes that gained data availability
        """
        print(
            f"\nPropagating data to ancestor nodes using method: {aggregation_method}"
        )

        initial_count = len(self.data_available)
        nodes_updated = 0

        # Process nodes in reverse topological order (leaves first, then parents)
        # Use BFS from leaves upward
        processed = set()
        to_process = deque(self.leaf_nodes)

        # Track descendant counts for each node
        descendant_counts = defaultdict(lambda: {"with_data": 0, "total": 0})

        while to_process:
            current = to_process.popleft()

            if current in processed:
                continue

            processed.add(current)

            # If this is a leaf, count it
            if current in self.leaf_nodes:
                parent = self.parent_map.get(current)
                if parent and parent != current:
                    descendant_counts[parent]["total"] += 1
                    if current in self.data_available:
                        descendant_counts[parent]["with_data"] += 1

                    # Add parent to processing queue
                    if parent not in processed:
                        to_process.append(parent)
            else:
                # This is an internal node - aggregate from children
                if current in self.children_map:
                    total = 0
                    with_data = 0

                    for child in self.children_map[current]:
                        if child in self.leaf_nodes:
                            total += 1
                            if child in self.data_available:
                                with_data += 1
                        else:
                            # Get counts from this internal child
                            total += descendant_counts[child]["total"]
                            with_data += descendant_counts[child]["with_data"]

                    descendant_counts[current]["total"] = total
                    descendant_counts[current]["with_data"] = with_data

                    # Decide if this node should be marked as having data
                    should_have_data = False

                    if aggregation_method == "any":
                        should_have_data = with_data > 0
                    elif aggregation_method == "count":
                        should_have_data = (
                            with_data >= 1
                        )  # At least one descendant
                    elif aggregation_method == "majority":
                        should_have_data = (
                            total > 0 and (with_data / total) > 0.5
                        )
                    elif aggregation_method == "summary":
                        should_have_data = with_data > 0

                    # Update data availability
                    if should_have_data and current not in self.data_available:
                        self.data_available.add(current)
                        nodes_updated += 1

                        # Create aggregated metadata
                        if current not in self.metadata_map:
                            self.metadata_map[current] = {}

                        self.metadata_map[current][
                            "descendant_count_with_data"
                        ] = with_data
                        self.metadata_map[current]["descendant_count_total"] = (
                            total
                        )
                        self.metadata_map[current]["data_source"] = "propagated"

                # Add parent to processing queue
                parent = self.parent_map.get(current)
                if parent and parent != current and parent not in processed:
                    to_process.append(parent)

        print(f"Propagation complete:")
        print(f"  Initial nodes with data: {initial_count:,}")
        print(f"  Nodes updated: {nodes_updated:,}")
        print(f"  Final nodes with data: {len(self.data_available):,}")

        # Clear cache since data availability changed
        self.cache.clear()

        return nodes_updated

    def find_nearest_leaf_with_data(
        self, tax_id, max_distance=10, priority_override=None
    ):
        """
        Find nearest leaf nodes with data (useful for finding related species).
        This searches both up and down the tree to find leaf relatives.

        Args:
            tax_id: Starting taxonomy ID
            max_distance: Maximum taxonomic distance to search
            priority_override: Optional list of column names to override default priority_columns

        Returns:
            dict with 'leaves' (list of leaf nodes with data) and 'distance' info
        """
        if tax_id in self.data_available and tax_id in self.leaf_nodes:
            return {
                "leaves": [self._create_node_info(tax_id, 0)],
                "self_is_leaf_with_data": True,
            }

        priority_cols = (
            priority_override
            if priority_override is not None
            else self.priority_columns
        )

        # BFS to find nearest leaves with data
        # We'll search by going up to ancestors and down to their descendants
        nearest_leaves = []
        min_distance_found = float("inf")

        # Strategy: Walk up the tree, and at each level, search descendants for leaves with data
        current = tax_id
        distance_to_ancestor = 0
        visited_ancestors = set()

        while (
            current in self.parent_map and distance_to_ancestor <= max_distance
        ):
            if current in visited_ancestors:
                break
            visited_ancestors.add(current)

            # Search descendants of this ancestor for leaves with data
            leaves_here = self._find_leaves_with_data_in_subtree(
                current,
                max_depth=max_distance - distance_to_ancestor,
                priority_cols=priority_cols,
            )

            # Calculate total distance (up to ancestor + down to leaf)
            for leaf_info in leaves_here:
                total_distance = distance_to_ancestor + leaf_info["distance"]
                if (
                    total_distance <= max_distance
                    and total_distance <= min_distance_found
                ):
                    leaf_info["distance"] = total_distance
                    if total_distance < min_distance_found:
                        nearest_leaves = [leaf_info]
                        min_distance_found = total_distance
                    elif total_distance == min_distance_found:
                        nearest_leaves.append(leaf_info)

            # Move to parent
            parent = self.parent_map.get(current)
            if parent == current:  # Root
                break
            current = parent
            distance_to_ancestor += 1

        # Prioritize and return
        prioritized_leaves = self._prioritize_candidates(
            nearest_leaves, priority_cols
        )

        return {"leaves": prioritized_leaves, "self_is_leaf_with_data": False}

    def _find_leaves_with_data_in_subtree(
        self, root_id, max_depth, priority_cols
    ):
        """
        Find all leaves with data in the subtree rooted at root_id.

        Args:
            root_id: Root of subtree to search
            max_depth: Maximum depth to search
            priority_cols: Priority columns for sorting

        Returns:
            List of leaf node info dicts with distance from root_id
        """
        if root_id in self.leaf_nodes and root_id in self.data_available:
            return [self._create_node_info(root_id, 0)]

        leaves = []
        queue = deque([(root_id, 0)])
        visited = set()

        while queue:
            node_id, depth = queue.popleft()

            if node_id in visited or depth > max_depth:
                continue
            visited.add(node_id)

            # Check if this is a leaf with data
            if node_id in self.leaf_nodes and node_id in self.data_available:
                leaves.append(self._create_node_info(node_id, depth))

            # Add children to queue
            if depth < max_depth and node_id in self.children_map:
                for child in self.children_map[node_id]:
                    if child not in visited:
                        queue.append((child, depth + 1))

        return leaves

    def find_nearest_with_data(
        self, tax_id, max_distance=10, priority_override=None
    ):
        """
        Find nearest taxonomic neighbor(s) with available data.

        Args:
            tax_id: Starting taxonomy ID
            max_distance: Maximum taxonomic distance to search
            priority_override: Optional list of column names to override default priority_columns

        Returns:
            dict with 'ancestor', 'descendants', and 'distance' info
        """
        cache_key = (
            tax_id,
            max_distance,
            tuple(priority_override or self.priority_columns),
        )
        if cache_key in self.cache:
            return self.cache[cache_key]

        if tax_id in self.data_available:
            result = {
                "ancestor": self._create_node_info(tax_id, 0),
                "descendants": [],
                "self_has_data": True,
            }
            self.cache[cache_key] = result
            return result

        priority_cols = (
            priority_override
            if priority_override is not None
            else self.priority_columns
        )

        # BFS to find nearest ancestor with data
        nearest_ancestor = self._find_nearest_ancestor_with_data(
            tax_id, max_distance, priority_cols
        )

        # BFS to find nearest descendants with data
        nearest_descendants = self._find_nearest_descendants_with_data(
            tax_id, max_distance, priority_cols
        )

        result = {
            "ancestor": nearest_ancestor,
            "descendants": nearest_descendants,
            "self_has_data": False,
        }

        self.cache[cache_key] = result
        return result

    def find_nearest_with_data_batch(
        self,
        tax_ids,
        max_distance=10,
        priority_override=None,
        leaves_only=False,
        return_stats=False,
    ):
        """
        Find nearest neighbors with data for multiple tax_ids .

        Args:
            tax_ids: List of taxonomy IDs to query
            max_distance: Maximum taxonomic distance to search
            priority_override: Optional list of column names to override default priority_columns
            leaves_only: If True, only process leaf nodes
            return_stats: If True, return statistics about the batch query

        Returns:
            dict mapping tax_id -> result dict (same format as find_nearest_with_data)
            If return_stats=True, also returns stats dict
        """
        # Filter to leaves only if requested
        if leaves_only:
            tax_ids = [tid for tid in tax_ids if tid in self.leaf_nodes]

        results = {}
        stats = {
            "total_queried": len(tax_ids),
            "self_has_data": 0,
            "ancestor_found": 0,
            "descendant_found": 0,
            "no_match_found": 0,
            "distance_distribution": defaultdict(int),
            "unmatched_tax_ids": [],
        }

        for tax_id in tax_ids:
            result = self.find_nearest_with_data(
                tax_id, max_distance, priority_override
            )
            results[tax_id] = result

            if return_stats:
                if result["self_has_data"]:
                    stats["self_has_data"] += 1
                    stats["distance_distribution"][0] += 1
                else:
                    has_match = False
                    if result["ancestor"]:
                        stats["ancestor_found"] += 1
                        stats["distance_distribution"][
                            result["ancestor"]["distance"]
                        ] += 1
                        has_match = True
                    if result["descendants"]:
                        stats["descendant_found"] += 1
                        min_desc_dist = min(
                            d["distance"] for d in result["descendants"]
                        )
                        stats["distance_distribution"][min_desc_dist] += 1
                        has_match = True
                    if not has_match:
                        stats["no_match_found"] += 1
                        stats["unmatched_tax_ids"].append(tax_id)

        if return_stats:
            return results, stats
        return results

    def find_nearest_leaf_with_data_batch(
        self,
        tax_ids,
        max_distance=10,
        priority_override=None,
        leaves_only=False,
        return_stats=False,
    ):
        """
        Find nearest leaf nodes with data for multiple tax_ids .

        Args:
            tax_ids: List of taxonomy IDs to query
            max_distance: Maximum taxonomic distance to search
            priority_override: Optional list of column names to override default priority_columns
            leaves_only: If True, only process leaf nodes (filter input)
            return_stats: If True, return statistics about the batch query

        Returns:
            dict mapping tax_id -> result dict (same format as find_nearest_leaf_with_data)
            If return_stats=True, also returns stats dict
        """
        # Filter to leaves only if requested
        if leaves_only:
            tax_ids = [tid for tid in tax_ids if tid in self.leaf_nodes]

        results = {}
        stats = {
            "total_queried": len(tax_ids),
            "self_is_leaf_with_data": 0,
            "leaves_found": 0,
            "no_leaves_found": 0,
            "distance_distribution": defaultdict(int),
            "unmatched_tax_ids": [],
            "total_leaves_returned": 0,
        }

        for tax_id in tax_ids:
            result = self.find_nearest_leaf_with_data(
                tax_id, max_distance, priority_override
            )
            results[tax_id] = result

            if return_stats:
                if result["self_is_leaf_with_data"]:
                    stats["self_is_leaf_with_data"] += 1
                    stats["distance_distribution"][0] += 1
                    stats["total_leaves_returned"] += 1
                elif result["leaves"]:
                    stats["leaves_found"] += 1
                    min_distance = min(
                        leaf["distance"] for leaf in result["leaves"]
                    )
                    stats["distance_distribution"][min_distance] += 1
                    stats["total_leaves_returned"] += len(result["leaves"])
                else:
                    stats["no_leaves_found"] += 1
                    stats["unmatched_tax_ids"].append(tax_id)

        if return_stats:
            return results, stats
        return results

    def find_nearest_with_data_unified(
        self,
        tax_id,
        max_rank="genus",
        priority_override=None,
        include_leaves=True,
        include_ancestors=True,
    ):
        """
        Unified method combining ancestor and leaf search, constrained by max_rank.

        Args:
            tax_id: Starting taxonomy ID
            max_rank: Maximum rank to traverse ('genus', 'family', etc.). Search stops at this rank.
            priority_override: Optional list of column names to override default priority_columns
            include_leaves: If True, search for nearest leaf relatives
            include_ancestors: If True, search for nearest ancestors

        Returns:
            dict with:
                - 'query_info': Info about the query tax_id
                - 'ancestor': Nearest ancestor with data (or None)
                - 'leaves': List of nearest leaf relatives with data
                - 'self_has_data': Boolean
                - 'max_rank_reached': Boolean indicating if search was limited by rank
        """
        priority_cols = (
            priority_override
            if priority_override is not None
            else self.priority_columns
        )

        # Get query info
        query_info = {
            "tax_id": tax_id,
            "name": self.name_map.get(tax_id),
            "rank": self.rank_map.get(tax_id),
            "is_leaf": tax_id in self.leaf_nodes,
        }

        # Check if self has data
        if tax_id in self.data_available:
            result = {
                "query_info": query_info,
                "ancestor": self._create_node_info(tax_id, 0),
                "leaves": [self._create_node_info(tax_id, 0)]
                if tax_id in self.leaf_nodes
                else [],
                "self_has_data": True,
                "max_rank_reached": False,
            }
            return result

        result = {
            "query_info": query_info,
            "ancestor": None,
            "leaves": [],
            "self_has_data": False,
            "max_rank_reached": False,
        }

        # Find the tax_id of the max_rank node in lineage (search boundary)
        max_rank_tax_id = self._find_rank_boundary(tax_id, max_rank)

        # Search for ancestor within rank boundary
        if include_ancestors:
            result["ancestor"], rank_reached = (
                self._find_nearest_ancestor_with_data_constrained(
                    tax_id, max_rank_tax_id, priority_cols
                )
            )
            result["max_rank_reached"] = (
                result["max_rank_reached"] or rank_reached
            )

        # Search for leaf relatives within rank boundary
        if include_leaves:
            result["leaves"], rank_reached = (
                self._find_nearest_leaf_with_data_constrained(
                    tax_id, max_rank_tax_id, priority_cols
                )
            )
            result["max_rank_reached"] = (
                result["max_rank_reached"] or rank_reached
            )

        return result

    def find_nearest_with_data_unified_batch(
        self,
        tax_ids,
        max_rank="genus",
        priority_override=None,
        include_leaves=True,
        include_ancestors=True,
        return_stats=False,
    ):
        """
        Batch version of unified search with rank constraints.

        Args:
            tax_ids: List of taxonomy IDs to query
            max_rank: Maximum rank to traverse ('genus', 'family', etc.)
            priority_override: Optional list of column names to override default priority_columns
            include_leaves: If True, search for nearest leaf relatives
            include_ancestors: If True, search for nearest ancestors
            return_stats: If True, return statistics about the batch query

        Returns:
            dict mapping tax_id -> result dict
            If return_stats=True, also returns stats dict
        """
        results = {}
        stats = {
            "total_queried": len(tax_ids),
            "self_has_data": 0,
            "ancestor_found": 0,
            "leaves_found": 0,
            "no_match_found": 0,
            "max_rank_limited": 0,
            "unmatched_tax_ids": [],
        }
        print(f"Search constraints: max_rank='{max_rank}'\n")

        for tax_id in tax_ids:
            result = self.find_nearest_with_data_unified(
                tax_id,
                max_rank,
                priority_override,
                include_leaves,
                include_ancestors,
            )
            results[tax_id] = result

            if return_stats:
                if result["self_has_data"]:
                    stats["self_has_data"] += 1
                else:
                    has_match = False

                    if result["ancestor"]:
                        stats["ancestor_found"] += 1
                        has_match = True

                    if result["leaves"]:
                        stats["leaves_found"] += 1
                        has_match = True

                    if not has_match:
                        stats["no_match_found"] += 1
                        stats["unmatched_tax_ids"].append(tax_id)

                    if result["max_rank_reached"]:
                        stats["max_rank_limited"] += 1

        if return_stats:
            return results, stats
        return results

    def _find_rank_boundary(self, tax_id, max_rank):
        """
        Find the tax_id of the ancestor at max_rank level.
        Returns None if max_rank is not found in lineage.
        """
        if not max_rank:
            return None

        current = tax_id
        visited = set()

        while current in self.parent_map and current not in visited:
            visited.add(current)

            if self.rank_map.get(current) == max_rank:
                return current

            parent = self.parent_map.get(current)
            if parent == current:  # Root
                break
            current = parent

        return None

    def _find_nearest_ancestor_with_data_constrained(
        self, tax_id, max_rank_tax_id, priority_cols
    ):
        """
        Find nearest ancestor with data, constrained by rank boundary.
        Returns (node_info, rank_reached_bool)
        """
        current = tax_id
        rank_reached = False

        while current in self.parent_map:
            parent = self.parent_map[current]

            # Check if we've reached the rank boundary
            if max_rank_tax_id and parent == max_rank_tax_id:
                rank_reached = True
                if parent in self.data_available:
                    return self._create_node_info(
                        parent, 0
                    ), rank_reached  # Distance not used
                break

            if parent in self.data_available:
                return self._create_node_info(parent, 0), rank_reached

            if parent == current:  # Root
                break
            current = parent

        return None, rank_reached

    def _find_nearest_leaf_with_data_constrained(
        self, tax_id, max_rank_tax_id, priority_cols
    ):
        """
        Find nearest leaf nodes with data, constrained by rank boundary.
        Returns (list of leaf nodes, rank_reached_bool)
        """
        nearest_leaves = []
        rank_reached = False

        # Walk up to max_rank, then search descendants for leaves
        current = tax_id
        visited_ancestors = set()

        while current in self.parent_map:
            if current in visited_ancestors:
                break
            visited_ancestors.add(current)

            # If at max_rank boundary, search its subtree and stop
            if max_rank_tax_id and current == max_rank_tax_id:
                rank_reached = True
                leaves_here = self._find_leaves_with_data_in_subtree(
                    current, priority_cols
                )
                nearest_leaves.extend(leaves_here)
                break

            # Search descendants for leaves
            leaves_here = self._find_leaves_with_data_in_subtree(
                current, priority_cols
            )
            nearest_leaves.extend(leaves_here)

            # Move to parent
            parent = self.parent_map.get(current)
            if parent == current:  # Root
                break
            current = parent

        # Prioritize and return unique nearest leaves
        prioritized_leaves = self._prioritize_candidates(
            list(set(nearest_leaves)), priority_cols
        )
        return prioritized_leaves, rank_reached

    def _find_leaves_with_data_in_subtree(self, root_id, priority_cols):
        """
        Find all leaves with data in the subtree rooted at root_id (optimized BFS).
        """
        leaves = []
        queue = deque([root_id])
        visited = set()

        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)

            if node_id in self.leaf_nodes and node_id in self.data_available:
                leaves.append(
                    self._create_node_info(node_id, 0)
                )  # Distance not used

            if node_id in self.children_map:
                queue.extend(self.children_map[node_id])

        return leaves

    def get_lineage_path(self, tax_id):
        """Get full lineage path from root to tax_id"""
        lineage = []
        current = tax_id
        visited = set()

        while current in self.parent_map and current not in visited:
            visited.add(current)
            lineage.append(
                {
                    "tax_id": current,
                    "name": self.name_map.get(current),
                    "rank": self.rank_map.get(current),
                    "has_data": current in self.data_available,
                }
            )

            parent = self.parent_map[current]
            if parent == current:  # Root
                break
            current = parent

        return list(reversed(lineage))

    def find_common_ancestor(self, tax_id1, tax_id2):
        """Find lowest common ancestor of two tax_ids"""
        lineage1 = {
            node["tax_id"]: i
            for i, node in enumerate(self.get_lineage_path(tax_id1))
        }
        lineage2_path = self.get_lineage_path(tax_id2)

        # Find first common node in lineage2 that's also in lineage1
        for node in lineage2_path:
            if node["tax_id"] in lineage1:
                return node

        return None

    def get_tree_stats(self):
        """Get comprehensive statistics about the tree structure and data availability"""
        stats = {
            "total_nodes": len(self.parent_map),
            "leaf_nodes": len(self.leaf_nodes),
            "internal_nodes": len(self.parent_map) - len(self.leaf_nodes),
            "nodes_with_data": len(self.data_available),
            "leaves_with_data": len(self.leaf_nodes & self.data_available),
            "leaves_without_data": len(self.leaf_nodes - self.data_available),
            "internal_nodes_with_data": len(
                self.data_available - self.leaf_nodes
            ),
            "coverage_percent_all": (
                len(self.data_available) / len(self.parent_map) * 100
            )
            if self.parent_map
            else 0,
            "coverage_percent_leaves": (
                len(self.leaf_nodes & self.data_available)
                / len(self.leaf_nodes)
                * 100
            )
            if self.leaf_nodes
            else 0,
            "cached_queries": len(self.cache),
            "priority_columns": self.priority_columns,
        }

        # Rank distribution
        rank_dist = defaultdict(int)
        rank_with_data = defaultdict(int)
        for tax_id in self.parent_map:
            rank = self.rank_map.get(tax_id, "unknown")
            rank_dist[rank] += 1
            if tax_id in self.data_available:
                rank_with_data[rank] += 1

        stats["rank_distribution"] = dict(rank_dist)
        stats["rank_with_data_distribution"] = dict(rank_with_data)

        return stats

    def print_stats(self):
        """Pretty print tree statistics"""
        stats = self.get_tree_stats()

        print("TAXONOMY TREE STATISTICS")
        print(f"\nTree Structure:")
        print(f"  Total nodes:          {stats['total_nodes']:,}")
        print(f"  Leaf nodes:           {stats['leaf_nodes']:,}")
        print(f"  Internal nodes:       {stats['internal_nodes']:,}")

        print(f"\nData Availability:")
        print(f"  Nodes with data:      {stats['nodes_with_data']:,}")
        print(f"  Leaves with data:     {stats['leaves_with_data']:,}")
        print(f"  Leaves without data:  {stats['leaves_without_data']:,}")
        print(f"  Internal nodes w/data:{stats['internal_nodes_with_data']:,}")

        print(f"\nCoverage:")
        print(f"  All nodes:            {stats['coverage_percent_all']:.2f}%")
        print(
            f"  Leaf nodes only:      {stats['coverage_percent_leaves']:.2f}%"
        )

        print(f"\nCache:")
        print(f"  Cached queries:       {stats['cached_queries']:,}")

        if stats["priority_columns"]:
            print(f"\nPriority columns: {', '.join(stats['priority_columns'])}")

        print(f"\nTop ranks by count:")
        sorted_ranks = sorted(
            stats["rank_distribution"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        for rank, count in sorted_ranks:
            with_data = stats["rank_with_data_distribution"].get(rank, 0)
            pct = (with_data / count * 100) if count > 0 else 0
            print(
                f"  {rank:20s}: {count:8,} total ({with_data:6,} with data, {pct:5.1f}%)"
            )

        print("=" * 60)


# print("TaxonomyTree class defined for neighbor finding")
