# %%
import io
import os
import re
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@pd.api.extensions.register_dataframe_accessor("emt")
class EmtAccessor:
    """
    Accessor for working with BTS ASCII format .emt files.

    This class provides methods for reading and working with BTS ASCII format .emt files.
    It can read the file and return metadata and dataframes for different sections.

    Example usage:
    --------------
    # Read a specific section from the file
    df = pd.DataFrame.emt.from_emt('data.emt', name='TimeSequences')

    # Get metadata associated with the DataFrame
    metadata = df.emt.metadata

    # Read all sections from the file
    all_sections = pd.DataFrame.emt.get_all_sections('data.emt')
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @staticmethod
    def read_file(file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Read BTS ASCII format .emt file and return metadata and dataframes.

        Parameters:
        -----------
        file_path : str
            Path to the .emt file

        Returns:
        --------
        dict
            Dictionary with section names as keys and dictionaries containing metadata and dataframes as values
        """
        with open(file_path, "r") as file:
            content = file.read()

        # Split the file content by the BTS ASCII format header to separate sections
        sections = re.split(r"BTS ASCII format\n", content)
        sections = [s for s in sections if s.strip()]  # Remove empty sections

        result = {}

        for section in sections:
            metadata = {}

            # Extract metadata using regex
            type_match = re.search(r"Type:\s+(.+)", section)
            if type_match:
                metadata["Type"] = type_match.group(1).strip()

            unit_match = re.search(r"Measure unit:\s+(.+)", section)
            if unit_match:
                metadata["Measure_unit"] = unit_match.group(1).strip()

            if "Point 1D tracks" in metadata.get("Type", ""):
                section_name = "1DPointTracks"
                tracks_match = re.search(r"Tracks:\s+(\d+)", section)
                if tracks_match:
                    metadata["Tracks"] = int(tracks_match.group(1))

                freq_match = re.search(r"Frequency:\s+(\d+)\s+Hz", section)
                if freq_match:
                    metadata["Frequency"] = int(freq_match.group(1))

                frames_match = re.search(r"Frames:\s+(\d+)", section)
                if frames_match:
                    metadata["Frames"] = int(frames_match.group(1))

                start_time_match = re.search(r"Start time:\s+(.+)", section)
                if start_time_match:
                    metadata["Start_time"] = float(start_time_match.group(1))

            elif "Time sequences" in metadata.get("Type", ""):
                section_name = "TimeSequences"
                seq_match = re.search(r"Sequences:\s+(\d+)", section)
                if seq_match:
                    metadata["Sequences"] = int(seq_match.group(1))

            elif "Scalar tracks" in metadata.get("Type", ""):
                section_name = "ScalarTracks"
                tracks_match = re.search(r"Tracks:\s+(\d+)", section)
                if tracks_match:
                    metadata["Tracks"] = int(tracks_match.group(1))

                freq_match = re.search(r"Frequency:\s+(\d+)\s+Hz", section)
                if freq_match:
                    metadata["Frequency"] = int(freq_match.group(1))

                frames_match = re.search(r"Frames:\s+(\d+)", section)
                if frames_match:
                    metadata["Frames"] = int(frames_match.group(1))

                start_time_match = re.search(r"Start time:\s+(.+)", section)
                if start_time_match:
                    metadata["Start_time"] = float(start_time_match.group(1))

            elif "Point 3D tracks" in metadata.get("Type", ""):
                section_name = "3DPointTracks"
                tracks_match = re.search(r"Tracks:\s+(\d+)", section)
                if tracks_match:
                    metadata["Tracks"] = int(tracks_match.group(1))

                freq_match = re.search(r"Frequency:\s+(\d+)\s+Hz", section)
                if freq_match:
                    metadata["Frequency"] = int(freq_match.group(1))

                frames_match = re.search(r"Frames:\s+(\d+)", section)
                if frames_match:
                    metadata["Frames"] = int(frames_match.group(1))

                start_time_match = re.search(r"Start time:\s+(.+)", section)
                if start_time_match:
                    metadata["Start_time"] = float(start_time_match.group(1))

            elif "Scalar values" in metadata.get("Type", ""):
                section_name = "ScalarValues"

                values = re.search(r"Values:\s+(\d+)", section)
                if values:
                    metadata["Values"] = int(values.group(1))

            elif "Volume tracks" in metadata.get("Type", ""):
                section_name = "VolumeTracks"
                tracks_match = re.search(r"Tracks:\s+(\d+)", section)
                if tracks_match:
                    metadata["Tracks"] = int(tracks_match.group(1))

                freq_match = re.search(r"Frequency:\s+(\d+)\s+Hz", section)
                if freq_match:
                    metadata["Frequency"] = int(freq_match.group(1))

                frames_match = re.search(r"Frames:\s+(\d+)", section)
                if frames_match:
                    metadata["Frames"] = int(frames_match.group(1))

                start_time_match = re.search(r"Start time:\s+(.+)", section)
                if start_time_match:
                    metadata["Start_time"] = float(start_time_match.group(1))

            else:
                # For any other section type
                section_name = f"Section_{len(result) + 1}"

            # Find the start of the data table
            if section_name == "ScalarValues":
                table_start = section.find("Scalar Insp Mean Time")
            elif section_name == "TimeSequences":
                table_start = section.find(" Item")
            elif section_name == "ScalarTracks":
                table_start = section.find(" Frame")
            elif section_name == "3DPointTracks":
                table_start = section.find(" Frame")
            elif section_name == "1DPointTracks":
                table_start = section.find(" Frame")
            elif section_name == "VolumeTracks":
                table_start = section.find(" Frame")
            else:
                table_start = -1

            if table_start != -1:
                # Extract the table part
                table_text = section[table_start:]

                # Read the table using pandas
                df = pd.read_csv(io.StringIO(table_text), sep="\t", skipinitialspace=True)

                # Clean up column names
                df.columns = [col.strip() for col in df.columns]

                # Remove empty "Unnamed:" columns
                df = df.loc[:, ~df.columns.str.contains("^Unnamed:")]

                # Store metadata in the DataFrame
                for key, value in metadata.items():
                    setattr(df, key, value)

                # Also store complete metadata dictionary
                df.attrs["emt_metadata"] = metadata

                # Special handling of 3D point tracks
                if section_name == "3DPointTracks":
                    # Fill empty using interpolation
                    df = df.interpolate(method="linear", limit_direction="both")

                result[section_name] = {"metadata": metadata, "dataframe": df}

        return result

    @staticmethod
    def determine_type_from_filename(file_path: str) -> Optional[str]:
        """
        Determine the section type based on the filename.

        Parameters:
        -----------
        file_path : str
            Path to the .emt file

        Returns:
        --------
        str or None
            Section name ('TimeSequences', 'ScalarTracks', etc.) or None if can't determine
        """
        # Get the base filename without extension
        basename = os.path.basename(file_path)
        filename, _ = os.path.splitext(basename)

        # Map of filenames to section types
        filename_map = {
            "Time Sequences": "TimeSequences",
            "TimeSequences": "TimeSequences",
            "Scalars": "ScalarValues",
            "Scalar Values": "ScalarValues",
            "ScalarValues": "ScalarValues",
            "Scalar Tracks": "ScalarTracks",
            "ScalarTracks": "ScalarTracks",
            "3D Point Tracks": "3DPointTracks",
            "3DPointTracks": "3DPointTracks",
            "1D Point Tracks": "1DPointTracks",
            "1DPointTracks": "1DPointTracks",
            "Volume Tracks": "VolumeTracks",
            "VolumeTracks": "VolumeTracks",
        }

        # Check for exact match
        if filename in filename_map:
            return filename_map[filename]

        # Try removing spaces and checking again
        no_spaces = filename.replace(" ", "")
        for key, value in filename_map.items():
            if no_spaces == key.replace(" ", ""):
                return value

        return None

    @classmethod
    def from_emt(cls, file_path: str, name: str = None) -> pd.DataFrame:
        """
        Create a DataFrame from an EMT file.

        Parameters:
        -----------
        file_path : str
            Path to the .emt file
        name : str, optional
            Name of the section to return (e.g., 'TimeSequences', 'ScalarTracks')
            If None, tries to determine from filename, then defaults to first section.

        Returns:
        --------
        pd.DataFrame
            DataFrame with EMT data and metadata as attributes
        """
        emt_data = cls.read_file(file_path)

        # If name is not specified, try to determine from filename
        if name is None:
            name = cls.determine_type_from_filename(file_path)

            print("Determined name: ", name)
            print("Available sections: ", emt_data.keys())

            # If still None, use the first section
            if name is None or name not in emt_data:
                first_key = next(iter(emt_data))
                return emt_data[first_key]["dataframe"]

        if name in emt_data:
            return emt_data[name]["dataframe"]
        else:
            available_sections = ", ".join(emt_data.keys())
            raise ValueError(f"Section '{name}' not found. Available sections: {available_sections}")

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get the metadata associated with this DataFrame.

        Returns:
        --------
        dict
            Metadata dictionary
        """
        return self._obj.attrs.get("emt_metadata", {})

    @property
    def is_3d_point_tracks(self) -> bool:
        """
        Check if the DataFrame represents 3D point tracks.

        Returns:
        --------
        bool
            True if this is a 3D point tracks DataFrame
        """
        return "3DPointTracks" in self.metadata.get("Type", "")


@pd.api.extensions.register_dataframe_accessor("emt3d")
class Emt3DAccessor(EmtAccessor):
    def __init__(self, pandas_obj):
        super().__init__(pandas_obj)

    @property
    def point_count(self) -> int:
        """
        Get the number of 3D points in the DataFrame.

        Returns:
        --------
        int
            Number of 3D points
        """
        return int(len([col for col in self._obj.columns if re.match(r"\d+\.(X|Y|Z)", col)]) / 3)

    def get_point(self, point: int) -> pd.DataFrame:
        """
        Get the data for a specific 3D point.

        Parameters:
        -----------
        point : int
            Point number (1-based)

        Returns:
        --------
        pd.DataFrame
            DataFrame with X, Y, Z coordinates for the point
        """
        if point < 1 or point > self.point_count:
            raise ValueError(f"Point number {point} is out of range")

        return self._obj[[f"{point}.X", f"{point}.Y", f"{point}.Z"]]

    def plot_point_in_time(self, point: int) -> go.Figure:
        """
        Plot the data for a specific 3D point's trajectory with color changing over time.

        Parameters:
        -----------
        point : int
            Point number (1-based)
        """
        # Extract data for the specific point
        x_data = self._obj[f"{point}.X"]
        y_data = self._obj[f"{point}.Y"]
        z_data = self._obj[f"{point}.Z"]
        time_data = self._obj["Time"]

        # Create the 3D line trace
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode="lines+markers",  # Show both line and markers
                    line=dict(
                        width=4,
                        color=time_data,  # Color based on time
                        colorscale="Viridis",  # Color gradient
                        showscale=True,  # Show color scale legend
                    ),
                    marker=dict(
                        size=1,
                        showscale=False,
                        color=time_data,
                        colorscale="Viridis",
                    ),
                )
            ]
        )

        # Customize layout for better visualization
        fig.update_layout(
            title=f"Trajectory of Point {point}",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=50),
        )

        return fig

    def plot_all_points_in_time(self):
        """
        Plot the data for all 3D points' trajectories with color changing over time.
        """
        fig = go.Figure()

        for point in range(1, self.point_count + 1):
            x_data = self._obj[f"{point}.X"]
            y_data = self._obj[f"{point}.Y"]
            z_data = self._obj[f"{point}.Z"]
            time_data = self._obj["Time"]

            # Add a trace for each point
            fig.add_trace(
                go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode="lines+markers",
                    line=dict(width=4, color=time_data, colorscale="Viridis", showscale=False),
                    marker=dict(
                        size=1,
                        showscale=False,
                        color=time_data,
                        colorscale="Viridis",
                    ),
                    name=f"Point {point}",
                )
            )

        # Customize layout for better visualization
        fig.update_layout(
            title="Trajectories of All 3D Points",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=50),
        )

        return fig

    def calculate_distance_for_point(self, point: int) -> pd.Series:
        """
        Calculate the distance covered by a specific 3D point.

        Parameters:
        -----------
        point : int
            Point number (1-based)

        Returns:
        --------
        pd.Series
            Series with distance covered at each time point
        """
        # Extract data for the specific point
        x_data = self._obj[f"{point}.X"]
        y_data = self._obj[f"{point}.Y"]
        z_data = self._obj[f"{point}.Z"]

        # Calculate distance using Euclidean norm
        distance = ((x_data.diff() ** 2 + y_data.diff() ** 2 + z_data.diff() ** 2) ** 0.5).cumsum()

        return distance

    def calculate_total_distance_for_point(self, point: int) -> float:
        """
        Calculate the total distance covered by a specific 3D point.

        Parameters:
        -----------
        point : int
            Point number (1-based)

        Returns:
        --------
        float
            Total distance covered by the point
        """
        distance = self.calculate_distance_for_point(point)
        return distance.iloc[-1]

    def calculate_total_distance_for_all_points(self) -> pd.Series:
        """
        Calculate the total distance covered by all 3D points.

        Returns:
        --------
        dict
            Dictionary with point numbers as keys and total distances as values
        """
        result = {}

        for point in range(1, self.point_count + 1):
            result[point] = self.calculate_total_distance_for_point(point)

        return pd.Series(result)

    def plot_total_distance_for_all_points(self) -> go.Figure:
        """
        Plot the total distance covered by all 3D points.
        """
        total_distances = self.calculate_total_distance_for_all_points()

        fig = px.bar(
            x=total_distances.index,
            y=total_distances.values,
            labels={"x": "Point", "y": "Total Distance"},
        )
        fig.update_layout(title="Total Distance Covered by 3D Points")
        return fig

    def plot_selection(self, selection_list: list = None):
        """
        Plot the data for selected 3D points' trajectories with color changing over time.

        Parameters:
        -----------
        selection_list : list, optional
            List of point numbers to plot (1-based). If None, all points are plotted.
        """

        records = []
        if not selection_list:
            selection_list = range(1, 82)
        for _, row in self._obj.iterrows():

            t = row["Time"]

            for point in selection_list:

                x = row.get(f"{point}.X", None)
                y = row.get(f"{point}.Y", None)
                z = row.get(f"{point}.Z", None)
                records.append({"Time": t, "Point": point, "X": x, "Y": y, "Z": z})
            break

        # Convert list of records into a DataFrame
        selection_df = pd.DataFrame(records)
        # Create an 3D scatter plot using Plotly Express
        fig = px.scatter_3d(
            selection_df,
            x="X",
            y="Y",
            z="Z",
            color="Point",
            title="3D Points",
            labels={"X": "X Axis", "Y": "Y Axis", "Z": "Z Axis"},
        )

        return fig

    def animate_points_in_time(self) -> go.Figure:
        """
        Animate the 3D points moving over time with improved performance.
        """
        # Create a list to store reshaped data
        records = []

        # Loop through each row (each time step)
        for _, row in self._obj.iterrows():
            t = row["Time"]
            # Loop over 81 points (assuming points are numbered from 1 to 81)
            for point in range(1, 82):
                x = row.get(f"{point}.X", None)
                y = row.get(f"{point}.Y", None)
                z = row.get(f"{point}.Z", None)
                records.append({"Time": t, "Point": point, "X": x, "Y": y, "Z": z})

        # Convert list of records into a DataFrame
        long_df = pd.DataFrame(records)

        # Create an animated 3D scatter plot using Plotly Express
        fig = px.scatter_3d(
            long_df,
            x="X",
            y="Y",
            z="Z",
            animation_frame="Time",
            animation_group="Point",
            color="Point",
            title="3D Animation of Moving Points",
            labels={"X": "X Axis", "Y": "Y Axis", "Z": "Z Axis"},
        )

        return fig

    def calculate_distance_between_points(self, point1: int, point2: int) -> pd.Series:
        """
        Calculate the distance between two 3D points.

        Parameters:
        -----------
        point1 : int
            First point number (1-based)
        point2 : int
            Second point number (1-based)

        Returns:
        --------
        pd.Series
            Series with distance between the points at each time point
        """
        # Extract data for the specific points
        x1_data = self._obj[f"{point1}.X"]
        y1_data = self._obj[f"{point1}.Y"]
        z1_data = self._obj[f"{point1}.Z"]

        x2_data = self._obj[f"{point2}.X"]
        y2_data = self._obj[f"{point2}.Y"]
        z2_data = self._obj[f"{point2}.Z"]

        # Calculate distance using Euclidean norm
        distance = ((x1_data - x2_data) ** 2 + (y1_data - y2_data) ** 2 + (z1_data - z2_data) ** 2) ** 0.5

        return distance

    def analyze_point_movements(self):
        """
        Analyze the movement of 3D points over time by calculating Euclidean distance
        between consecutive timestamps for each point.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the distances for each point at each timestamp.
        """
        import numpy as np
        import pandas as pd

        # Create a DataFrame to store the distances
        movement_data = pd.DataFrame(index=range(1, len(self._obj)))

        # Calculate Euclidean distance for each point between consecutive timestamps
        for point in range(1, self.point_count + 1):
            distances = []

            # Get the coordinate data for this point
            x_data = self._obj[f"{point}.X"]
            y_data = self._obj[f"{point}.Y"]
            z_data = self._obj[f"{point}.Z"]

            # Calculate distances between consecutive timestamps
            for i in range(1, len(self._obj)):
                # Previous position
                prev_x = x_data.iloc[i - 1]
                prev_y = y_data.iloc[i - 1]
                prev_z = z_data.iloc[i - 1]

                # Current position
                curr_x = x_data.iloc[i]
                curr_y = y_data.iloc[i]
                curr_z = z_data.iloc[i]

                # Euclidean distance: sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
                distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2 + (curr_z - prev_z) ** 2)

                distances.append(distance)

            # Add this point's distances to the DataFrame
            movement_data[f"Point {point}"] = distances

        # Add timestamp information
        if "timestamp" in self._obj.columns:
            movement_data["timestamp"] = self._obj["timestamp"].iloc[1:].values
        else:
            movement_data["timestamp_index"] = range(1, len(self._obj))

        return movement_data

    def visualize_movement_changes(self):
        """
        Visualize changes in movement for each point over time.

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure object containing the movement changes for each point.
        """
        import plotly.graph_objects as go

        # Get the movement data
        movement_data = self.analyze_point_movements()

        # Create a new figure
        fig = go.Figure()

        # Add a line for each point
        for point in range(1, self.point_count + 1):
            fig.add_trace(
                go.Scatter(
                    x=movement_data.index,
                    y=movement_data[f"Point {point}"],
                    mode="lines+markers",
                    name=f"Point {point}",
                    marker=dict(size=4),
                    line=dict(width=2),
                )
            )

        # Add threshold line for significant movements (adjustable)
        # Set this to a value that makes sense for your data
        threshold = movement_data.drop("timestamp", axis=1, errors="ignore").mean().mean() * 2
        fig.add_shape(
            type="line",
            x0=0,
            y0=threshold,
            x1=len(movement_data),
            y1=threshold,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            ),
        )

        # Annotate the threshold
        fig.add_annotation(
            x=len(movement_data) * 0.95,
            y=threshold * 1.1,
            text=f"Significant Movement Threshold ({threshold:.3f})",
            showarrow=False,
            font=dict(color="red"),
        )

        # Update layout
        fig.update_layout(
            title="Point Movement Analysis: Euclidean Distance Between Consecutive Timestamps",
            xaxis_title="Timestamp Index",
            yaxis_title="Distance (Euclidean)",
            legend_title="Points",
            hovermode="closest",
        )

        return fig

    def identify_significant_movements(self, threshold_factor=2.0):
        """
        Identify timestamps with significant movements for each point.

        Parameters
        ----------
        threshold_factor : float, optional
            Factor to multiply the average movement by to set the threshold.
            Default is 2.0, meaning movements twice the average are considered significant.

        Returns
        -------
        dict
            Dictionary where keys are point numbers and values are dictionaries containing:
            - 'threshold': The calculated threshold for significant movement.
            - 'timestamps': List of indices where movement exceeds the threshold.
            - 'distances': List of movement distances at the significant timestamps.
        """
        # Get the movement data
        movement_data = self.analyze_point_movements()

        # Calculate the average movement for each point
        avg_movements = movement_data.drop("timestamp", axis=1, errors="ignore").mean()

        # Find significant movements
        significant_movements = {}

        for point in range(1, self.point_count + 1):
            col_name = f"Point {point}"
            threshold = avg_movements[col_name] * threshold_factor

            # Find indices where movement exceeds threshold
            significant_indices = movement_data.index[movement_data[col_name] > threshold].tolist()

            # Store in the dictionary
            significant_movements[point] = {
                "threshold": threshold,
                "timestamps": significant_indices,
                "distances": [movement_data.loc[idx, col_name] for idx in significant_indices],
            }

        return significant_movements
