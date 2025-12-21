# %%
import os
import sys
import unittest

import numpy as np
import pandas as pd

from ml.datasets.gnn_vest_dataset import GNNVestDataset

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGNNVestDataset(unittest.TestCase):
    """Test cases for GNNVestDataset class with focus on distance calculations."""

    def setUp(self):
        """Set up simple test fixtures for precise distance verification."""
        # Create simple test data with known coordinates for easy distance calculation
        self.simple_df = self._create_simple_test_data()
        self.full_vest_df = self._create_full_vest_test_data()

    def _create_simple_test_data(self):
        """Create minimal 4-node test data with known coordinates."""
        # 4 nodes in a simple square pattern, 3 timesteps
        data = []

        # Timestep 0: Square with side length 1
        data.append(
            {
                "1.X": 0.0,
                "1.Y": 0.0,
                "1.Z": 0.0,  # Bottom-left
                "2.X": 1.0,
                "2.Y": 0.0,
                "2.Z": 0.0,  # Bottom-right
                "3.X": 1.0,
                "3.Y": 1.0,
                "3.Z": 0.0,  # Top-right
                "4.X": 0.0,
                "4.Y": 1.0,
                "4.Z": 0.0,  # Top-left
            }
        )

        # Timestep 1: Move all nodes up by 1 in Z
        data.append(
            {
                "1.X": 0.0,
                "1.Y": 0.0,
                "1.Z": 1.0,
                "2.X": 1.0,
                "2.Y": 0.0,
                "2.Z": 1.0,
                "3.X": 1.0,
                "3.Y": 1.0,
                "3.Z": 1.0,
                "4.X": 0.0,
                "4.Y": 1.0,
                "4.Z": 1.0,
            }
        )

        # Timestep 2: Expand square (scale by 2)
        data.append(
            {
                "1.X": 0.0,
                "1.Y": 0.0,
                "1.Z": 2.0,
                "2.X": 2.0,
                "2.Y": 0.0,
                "2.Z": 2.0,
                "3.X": 2.0,
                "3.Y": 2.0,
                "3.Z": 2.0,
                "4.X": 0.0,
                "4.Y": 2.0,
                "4.Z": 2.0,
            }
        )

        df = pd.DataFrame(data)
        df["Time"] = [0.0, 0.1, 0.2]
        df["recording_id"] = "simple_test"
        return df

    def _create_full_vest_test_data(self):
        """Create test data with 89 nodes for full vest topology testing."""
        np.random.seed(42)
        data = []

        for t in range(5):  # 5 timesteps
            row = {"Time": t * 0.1, "recording_id": "full_vest_test"}
            for node in range(1, 90):  # 89 nodes
                # Create predictable but varying coordinates
                row[f"{node}.X"] = node * 0.1 + t * 0.01
                row[f"{node}.Y"] = (node % 10) * 0.1 + t * 0.01
                row[f"{node}.Z"] = (node // 10) * 0.1 + t * 0.01
            data.append(row)

        return pd.DataFrame(data)

    def test_timestep_mode_distance_calculation(self):
        """Test distance calculations in timestep mode with simple known coordinates."""
        dataset = GNNVestDataset(self.simple_df, mode="timestep", validate_data=False)
        dataset.preprocess_data(apply_scaling=False)  # No scaling for exact distance verification
        graphs = dataset.create_graphs()

        # Should have 3 graphs (3 timesteps)
        self.assertEqual(len(graphs), 3)

        # Test first timestep (square with side length 1)
        graph0 = graphs[0]
        self.assertEqual(graph0.x.shape, (4, 3))  # 4 nodes, 3 coordinates

        # Get simplified edge connections for 4 nodes
        edge_index = graph0.edge_index
        edge_attr = graph0.edge_attr

        # Check some known distances
        coords = graph0.x.numpy()

        # Node positions at timestep 0:
        # Node 1: (0,0,0), Node 2: (1,0,0), Node 3: (1,1,0), Node 4: (0,1,0)
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])  # Node 1
        np.testing.assert_array_almost_equal(coords[1], [1.0, 0.0, 0.0])  # Node 2
        np.testing.assert_array_almost_equal(coords[2], [1.0, 1.0, 0.0])  # Node 3
        np.testing.assert_array_almost_equal(coords[3], [0.0, 1.0, 0.0])  # Node 4

        # Verify some edge distances
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()
            calculated_distance = edge_attr[i].item()

            # Calculate expected distance
            src_coords = coords[src_idx]
            dst_coords = coords[dst_idx]
            expected_distance = np.sqrt(np.sum((src_coords - dst_coords) ** 2))

            self.assertAlmostEqual(
                calculated_distance,
                expected_distance,
                places=5,
                msg=f"Distance mismatch for edge {src_idx}->{dst_idx}",
            )

    def test_sequence_mode_temporal_distance_changes(self):
        """Test sequence mode captures distance changes between timesteps."""
        dataset = GNNVestDataset(self.simple_df, mode="sequence", sequence_length=2, validate_data=False)
        dataset.preprocess_data(apply_scaling=False)
        graphs = dataset.create_graphs()

        # Should have 2 sequence graphs (timesteps 0-1 and 1-2)
        self.assertEqual(len(graphs), 2)

        # Test first sequence graph (timesteps 0 and 1)
        seq_graph = graphs[0]

        # Should have 8 nodes (4 nodes × 2 timesteps)
        self.assertEqual(seq_graph.x.shape[0], 8)

        # Get node features and edge information
        coords = seq_graph.x.numpy()
        edge_index = seq_graph.edge_index

        # Verify temporal edges exist (same point across timesteps)
        # Node 0 at t=0 should connect to Node 0 at t=1 (indices 0 and 4)
        temporal_edges_found = 0
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()

            # Check if this is a temporal edge (4 nodes apart = different timestep)
            if abs(src_idx - dst_idx) == 4:
                temporal_edges_found += 1

                # Calculate the distance change
                src_coords = coords[src_idx]
                dst_coords = coords[dst_idx]
                distance = np.sqrt(np.sum((src_coords - dst_coords) ** 2))

                # For our test data, temporal distance should be 1.0 (pure Z movement)
                self.assertAlmostEqual(
                    distance,
                    1.0,
                    places=5,
                    msg=f"Temporal distance incorrect for nodes {src_idx}->{dst_idx}",
                )

        # Should find temporal edges (bidirectional, so 8 total: 4 nodes × 2 directions)
        self.assertGreater(temporal_edges_found, 0, "No temporal edges found in sequence mode")

    def test_whole_file_mode_temporal_connections(self):
        """Test whole file mode creates proper temporal connections."""
        dataset = GNNVestDataset(self.simple_df, mode="whole_file", validate_data=False)
        dataset.preprocess_data(apply_scaling=False)
        graphs = dataset.create_graphs()

        # Should have 1 graph for entire file
        self.assertEqual(len(graphs), 1)

        graph = graphs[0]
        # Should have 12 nodes (4 nodes × 3 timesteps)
        self.assertEqual(graph.x.shape[0], 12)

        # Check for temporal connections
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        temporal_edges = []
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()

            # Temporal edges connect same point across consecutive timesteps (4 nodes apart)
            if abs(src_idx - dst_idx) == 4:
                temporal_edges.append((src_idx, dst_idx, edge_attr[i].item()))

        # Should have temporal edges between timesteps
        self.assertGreater(len(temporal_edges), 0, "No temporal edges found in whole_file mode")

        # Verify distances are calculated correctly for temporal edges
        coords = graph.x.numpy()
        for src_idx, dst_idx, calculated_dist in temporal_edges:
            src_coords = coords[src_idx]
            dst_coords = coords[dst_idx]
            expected_dist = np.sqrt(np.sum((src_coords - dst_coords) ** 2))

            self.assertAlmostEqual(
                calculated_dist,
                expected_dist,
                places=5,
                msg=f"Temporal edge distance incorrect: {src_idx}->{dst_idx}",
            )

    def test_full_vest_topology_edge_connections(self):
        """Test that full 89-node vest topology creates expected edge connections."""
        dataset = GNNVestDataset(self.full_vest_df, mode="timestep", validate_data=False)
        dataset.preprocess_data(apply_scaling=False)
        graphs = dataset.create_graphs()

        self.assertEqual(len(graphs), 5)  # 5 timesteps

        # Test first graph
        graph = graphs[0]
        self.assertEqual(graph.x.shape, (89, 3))  # 89 nodes, 3 coordinates

        # Check that we have the expected number of edges for vest topology
        # The vest has a specific connectivity pattern
        edge_index = graph.edge_index
        num_edges = edge_index.shape[1]

        # Should have bidirectional edges, so even number
        self.assertEqual(num_edges % 2, 0, "Should have even number of edges (bidirectional)")

        # Check that all edge distances are positive
        edge_attr = graph.edge_attr
        for i in range(len(edge_attr)):
            self.assertGreater(edge_attr[i].item(), 0, f"Edge {i} has non-positive distance")

        # Verify some known vest connections exist (based on the topology in the dataset)
        # Node 1 should connect to nodes 2, 8, 46, 47 (as per vest topology)
        node_1_connections = []
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src == 0:  # Node 1 (0-indexed)
                node_1_connections.append(dst + 1)  # Convert back to 1-indexed

        expected_connections = [2, 8, 46, 47]
        for expected in expected_connections:
            if expected <= 89:  # Make sure the connection exists in our topology
                self.assertIn(
                    expected,
                    node_1_connections,
                    f"Node 1 should connect to node {expected}",
                )

    def test_edge_attribute_consistency(self):
        """Test that edge attributes (distances) are consistent across modes."""
        # Create dataset with same data in different modes
        timestep_dataset = GNNVestDataset(self.simple_df, mode="timestep", validate_data=False)
        timestep_dataset.preprocess_data(apply_scaling=False)
        timestep_graphs = timestep_dataset.create_graphs()

        whole_file_dataset = GNNVestDataset(self.simple_df, mode="whole_file", validate_data=False)
        whole_file_dataset.preprocess_data(apply_scaling=False)
        whole_file_graphs = whole_file_dataset.create_graphs()

        # Compare spatial edges in first timestep
        timestep_graph = timestep_graphs[0]
        whole_file_graph = whole_file_graphs[0]

        # Get spatial edges from timestep graph
        ts_edges = {}
        ts_edge_index = timestep_graph.edge_index
        ts_edge_attr = timestep_graph.edge_attr

        for i in range(ts_edge_index.shape[1]):
            src = ts_edge_index[0, i].item()
            dst = ts_edge_index[1, i].item()
            dist = ts_edge_attr[i].item()
            ts_edges[(src, dst)] = dist

        # Get corresponding spatial edges from whole file graph (first timestep nodes: 0-3)
        wf_edges = {}
        wf_edge_index = whole_file_graph.edge_index
        wf_edge_attr = whole_file_graph.edge_attr

        for i in range(wf_edge_index.shape[1]):
            src = wf_edge_index[0, i].item()
            dst = wf_edge_index[1, i].item()

            # Only consider spatial edges within first timestep (nodes 0-3)
            if src < 4 and dst < 4:
                dist = wf_edge_attr[i].item()
                wf_edges[(src, dst)] = dist

        # Compare distances for common edges
        common_edges = set(ts_edges.keys()) & set(wf_edges.keys())
        self.assertGreater(len(common_edges), 0, "No common edges found between modes")

        for edge in common_edges:
            ts_dist = ts_edges[edge]
            wf_dist = wf_edges[edge]
            self.assertAlmostEqual(
                ts_dist,
                wf_dist,
                places=5,
                msg=f"Distance mismatch for edge {edge}: timestep={ts_dist}, whole_file={wf_dist}",
            )

    def test_scaling_effects_on_distances(self):
        """Test that scaling affects distances as expected."""
        # Test with scaling
        dataset_scaled = GNNVestDataset(self.simple_df, mode="timestep", scaler_type="standard", validate_data=False)
        dataset_scaled.preprocess_data(apply_scaling=True)
        graphs_scaled = dataset_scaled.create_graphs()

        # Test without scaling
        dataset_unscaled = GNNVestDataset(self.simple_df, mode="timestep", validate_data=False)
        dataset_unscaled.preprocess_data(apply_scaling=False)
        graphs_unscaled = dataset_unscaled.create_graphs()

        # Compare first graphs
        scaled_graph = graphs_scaled[0]
        unscaled_graph = graphs_unscaled[0]

        # Scaled coordinates should be different from unscaled
        scaled_coords = scaled_graph.x.numpy()
        unscaled_coords = unscaled_graph.x.numpy()

        # Coordinates should not be identical (scaling should change them)
        self.assertFalse(
            np.allclose(scaled_coords, unscaled_coords),
            "Scaling should change coordinate values",
        )

        # But the graph structure should be the same
        self.assertEqual(scaled_graph.edge_index.shape, unscaled_graph.edge_index.shape)

        # Edge distances should be different due to scaling
        scaled_distances = scaled_graph.edge_attr.numpy()
        unscaled_distances = unscaled_graph.edge_attr.numpy()

        self.assertFalse(
            np.allclose(scaled_distances, unscaled_distances),
            "Scaling should change edge distances",
        )

    def test_invalid_modes_and_parameters(self):
        """Test error handling for invalid modes and parameters."""
        # Invalid mode
        with self.assertRaises(ValueError):
            GNNVestDataset(self.simple_df, mode="invalid_mode")

        # Sequence mode without sequence_length
        with self.assertRaises(ValueError):
            GNNVestDataset(self.simple_df, mode="sequence", sequence_length=None)

        # Invalid dataframe type
        with self.assertRaises(AssertionError):
            GNNVestDataset("not_a_dataframe")

    def test_sequence_step_parameter(self):
        """Test sequence_step parameter reduces overlap in sequence graphs."""
        # Create test data with 6 timesteps
        test_data = []
        for i in range(6):
            test_data.append(
                {
                    "1.X": float(i),
                    "1.Y": 0.0,
                    "1.Z": 0.0,
                    "2.X": float(i + 1),
                    "2.Y": 0.0,
                    "2.Z": 0.0,
                }
            )
        df = pd.DataFrame(test_data)

        # Test with step=1 (default)
        dataset_step1 = GNNVestDataset(df, mode="sequence", sequence_length=3, sequence_step=1, validate_data=False)
        dataset_step1.preprocess_data(apply_scaling=False)
        graphs_step1 = dataset_step1.create_graphs()

        # Test with step=2 (skip every other)
        dataset_step2 = GNNVestDataset(df, mode="sequence", sequence_length=3, sequence_step=2, validate_data=False)
        dataset_step2.preprocess_data(apply_scaling=False)
        graphs_step2 = dataset_step2.create_graphs()

        # With 6 timesteps and sequence_length=3:
        # step=1: sequences starting at [0,1,2,3] = 4 graphs
        # step=2: sequences starting at [0,2,4] = 3 graphs (but 4 would exceed bounds, so 2 graphs)
        self.assertEqual(len(graphs_step1), 4)  # 6-3+1 = 4 sequences
        self.assertEqual(len(graphs_step2), 2)  # floor((6-3)/2) + 1 = 2 sequences

        # Verify that step=2 sequences start at correct positions
        self.assertEqual(graphs_step2[0].sequence_start, 0)  # First sequence: timesteps 0,1,2
        self.assertEqual(graphs_step2[1].sequence_start, 2)  # Second sequence: timesteps 2,3,4

    def test_no_scaling_option(self):
        """Test the 'none' scaler option preserves original coordinates."""
        # Create test data with known coordinates
        test_data = [
            {
                "1.X": 10.0,
                "1.Y": 20.0,
                "1.Z": 30.0,
                "2.X": 40.0,
                "2.Y": 50.0,
                "2.Z": 60.0,
            },
            {
                "1.X": 15.0,
                "1.Y": 25.0,
                "1.Z": 35.0,
                "2.X": 45.0,
                "2.Y": 55.0,
                "2.Z": 65.0,
            },
        ]
        df = pd.DataFrame(test_data)

        # Test with 'none' scaler
        dataset = GNNVestDataset(df, mode="timestep", scaler_type="none", validate_data=False)
        dataset.preprocess_data()

        # Verify coordinates are preserved exactly
        expected_first_timestep = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
        expected_second_timestep = [[15.0, 25.0, 35.0], [45.0, 55.0, 65.0]]

        np.testing.assert_array_almost_equal(
            dataset.node_features[0],
            expected_first_timestep,
            decimal=6,
            err_msg="First timestep coordinates should be preserved exactly",
        )
        np.testing.assert_array_almost_equal(
            dataset.node_features[1],
            expected_second_timestep,
            decimal=6,
            err_msg="Second timestep coordinates should be preserved exactly",
        )

        # Verify scaler is None
        self.assertIsNone(dataset.scaler, "Scaler should be None for 'none' scaler_type")

        # Create graphs and verify coordinates are still preserved
        graphs = dataset.create_graphs()
        self.assertEqual(len(graphs), 2)

        # Check first graph coordinates
        first_graph_coords = graphs[0].x.numpy()
        np.testing.assert_array_almost_equal(
            first_graph_coords,
            expected_first_timestep,
            decimal=6,
            err_msg="Graph coordinates should match original values exactly",
        )

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        dataset = GNNVestDataset(empty_df, validate_data=False)

        with self.assertRaises(ValueError):
            dataset.preprocess_data()
