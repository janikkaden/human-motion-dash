"""
Obstacle representation classes for spatial scene understanding
Handles static obstacles (walls, furniture) and dynamic participants (people)
"""

import numpy as np
import pandas as pd
import json
from scipy.spatial import ConvexHull


# ==================== CYLINDER OBSTACLE FOR PARTICIPANTS ====================
class CylinderObstacle:
    """
    Represents a cylindrical obstacle (e.g., a human participant)
    Used for detecting gaze intersections with other participants
    """
    def __init__(self, name, radius=300, height=1800):
        """
        Args:
            name: Identifier for the participant (e.g., "Helmet_2", "tim")
            radius: Cylinder radius in mm (default 300mm ~ 30cm shoulder width)
            height: Cylinder height in mm (default 1800mm ~ 1.8m person height)
        """
        self.name = name
        self.radius = radius
        self.height = height
        self.position_history = []

        # Pre-compute trigonometry for faster rendering
        self.segments = 16
        self.angles = np.linspace(0, 2*np.pi, self.segments)
        self.cos_angles = np.cos(self.angles)
        self.sin_angles = np.sin(self.angles)

    def update_position(self, x, y, z=0):
        """Update the current position of the cylinder"""
        self.current_position = np.array([x, y, z])
        self.position_history.append((x, y, z))

    def ray_intersection(self, ray_origin, ray_direction, max_distance=10000):
        """
        Calculate ray-cylinder intersection in 2D (XY plane)

        Uses quadratic formula to solve ray-cylinder intersection:
        - Ray: P(t) = O + t*D
        - Cylinder: (x-cx)² + (y-cy)² = r²
        - Results in: at² + bt + c = 0

        Args:
            ray_origin: 3D array [x, y, z] of ray start point
            ray_direction: 3D array [dx, dy, dz] of ray direction (normalized)
            max_distance: Maximum distance to check for intersection

        Returns:
            tuple: (intersection_point_3d, distance) or (None, None) if no intersection
        """
        if not hasattr(self, 'current_position'):
            return None, None

        ray_origin_2d = np.array([ray_origin[0], ray_origin[1]])
        ray_dir_2d = np.array([ray_direction[0], ray_direction[1]])

        ray_dir_2d_norm = np.linalg.norm(ray_dir_2d)
        if ray_dir_2d_norm < 1e-6:
            return None, None
        ray_dir_2d = ray_dir_2d / ray_dir_2d_norm

        oc = ray_origin_2d - self.current_position[:2]

        # Solve quadratic equation for ray-cylinder intersection
        a = np.dot(ray_dir_2d, ray_dir_2d)
        b = 2.0 * np.dot(oc, ray_dir_2d)
        c = np.dot(oc, oc) - self.radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return None, None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        # Select closest positive intersection
        t = None
        if t1 > 0 and t1 < max_distance:
            t = t1
        elif t2 > 0 and t2 < max_distance:
            t = t2

        if t is None or t <= 0:
            return None, None

        intersection_2d = ray_origin_2d + t * ray_dir_2d

        # Project back to 3D using original ray direction
        ray_dir_3d_norm = np.linalg.norm(ray_direction)
        if ray_dir_3d_norm < 1e-6:
            return None, None

        t_3d = t / ray_dir_3d_norm
        intersection_z = ray_origin[2] + t_3d * ray_direction[2]

        # Verify intersection is within cylinder height bounds
        z_base = self.current_position[2] if len(self.current_position) > 2 else 0
        if intersection_z < z_base or intersection_z > z_base + self.height:
            return None, None

        intersection_3d = np.array([intersection_2d[0], intersection_2d[1], intersection_z])
        distance = np.linalg.norm(intersection_3d - ray_origin)

        return intersection_3d, distance

    def get_vertices_3d(self):
        """Get 3D vertices for rendering the cylinder"""
        if not hasattr(self, 'current_position'):
            return [], []

        x = self.current_position[0] + self.radius * self.cos_angles
        y = self.current_position[1] + self.radius * self.sin_angles
        z_base = self.current_position[2] if len(self.current_position) > 2 else 0

        vertices_bottom = np.column_stack([x, y, np.full_like(x, z_base)])
        vertices_top = np.column_stack([x, y, np.full_like(x, z_base + self.height)])

        return vertices_bottom, vertices_top


# ==================== PARTICIPANT MANAGER ====================
class ParticipantManager:
    """
    Manages cylinder obstacles for other participants
    Allows gaze ray intersection detection with other people
    """
    def __init__(self, participant_names, cylinder_radius=300, cylinder_height=1800):
        """
        Args:
            participant_names: List of participant identifiers
            cylinder_radius: Radius of person cylinder in mm
            cylinder_height: Height of person cylinder in mm
        """
        self.cylinders = {}
        for name in participant_names:
            self.cylinders[name] = CylinderObstacle(name, cylinder_radius, cylinder_height)
        print(f"Created {len(self.cylinders)} participant cylinders: {participant_names}")

    def update_positions(self, data_row, candidate_cols):
        """
        Update all participant cylinder positions for current frame

        Args:
            data_row: Current row from trajectory DataFrame
            candidate_cols: Dictionary mapping participant names to their column prefixes
        """
        for name, cylinder in self.cylinders.items():
            if name in candidate_cols:
                col_prefix = candidate_cols[name]
                try:
                    x = data_row[f'{col_prefix}_TX']
                    y = data_row[f'{col_prefix}_TY']
                    z = data_row.get(f'{col_prefix}_TZ', 0)
                    if pd.notna(x) and pd.notna(y):
                        z_val = z if pd.notna(z) else 0
                        cylinder.update_position(x, y, z_val)
                except KeyError:
                    pass

    def check_gaze_intersections(self, ray_origin, ray_direction):
        """
        Check if gaze ray intersects with any participant cylinder

        Returns:
            tuple: (participant_name, intersection_point, distance) or (None, None, None)
        """
        closest_intersection = None
        closest_distance = float('inf')
        closest_participant = None

        for name, cylinder in self.cylinders.items():
            intersection, distance = cylinder.ray_intersection(ray_origin, ray_direction)
            if intersection is not None and distance < closest_distance:
                closest_intersection = intersection
                closest_distance = distance
                closest_participant = name

        return closest_participant, closest_intersection, closest_distance

    def draw_cylinders_3d(self, ax):
        """Draw all participant cylinders in 3D view"""
        for name, cylinder in self.cylinders.items():
            vertices_bottom, vertices_top = cylinder.get_vertices_3d()
            if len(vertices_bottom) > 0:
                # Render top and bottom circular edges
                bottom_points = np.array(list(vertices_bottom) + [vertices_bottom[0]])
                top_points = np.array(list(vertices_top) + [vertices_top[0]])

                ax.plot(bottom_points[:, 0], bottom_points[:, 1], bottom_points[:, 2],
                       'c-', alpha=0.6, linewidth=2)
                ax.plot(top_points[:, 0], top_points[:, 1], top_points[:, 2],
                       'c-', alpha=0.6, linewidth=2)

                # Render vertical support lines
                for i in range(0, len(vertices_bottom), 4):
                    ax.plot([vertices_bottom[i][0], vertices_top[i][0]],
                           [vertices_bottom[i][1], vertices_top[i][1]],
                           [vertices_bottom[i][2], vertices_top[i][2]],
                           'c-', alpha=0.6, linewidth=2)

                z_top = cylinder.current_position[2] + cylinder.height if len(cylinder.current_position) > 2 else cylinder.height
                ax.text(cylinder.current_position[0], cylinder.current_position[1],
                       z_top + 200, f'{name}', fontsize=10, color='cyan',
                       ha='center', weight='bold')


# ==================== STATIC OBSTACLE CLASSES ====================
class Obstacle:
    """Represents a 3D obstacle with base polygon and height"""
    def __init__(self, base_points, z_min, z_max, obstacle_type, source_file,
                 seat_height=None, backrest_edge=None):

        base_points = np.array(base_points)
        if len(base_points) == 4:
            base_points = self._order_rectangle_points(base_points)
        else:
            base_points = self._order_points_convex_hull(base_points)

        self.base_points = base_points
        self.z_min = 0.0
        self.z_max = float(z_max)
        self.obstacle_type = obstacle_type
        self.source_file = source_file
        self.height = self.z_max - self.z_min

        self.seat_height = seat_height
        self.is_chair = 'chair' in obstacle_type.lower()
        self.is_wall = 'wall' in obstacle_type.lower()
        self.backrest_edge = backrest_edge

    def _order_rectangle_points(self, points):
        """Order rectangle points counterclockwise"""
        centroid = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        return points[sorted_indices]

    def _order_points_convex_hull(self, points):
        """Order polygon points using convex hull"""
        if len(points) < 3:
            return points
        try:
            hull = ConvexHull(points)
            return points[hull.vertices]
        except:
            centroid = points.mean(axis=0)
            angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            return points[sorted_indices]

    def _get_backrest_edge_index(self):
        """
        Determine which edge index has the backrest.
        Returns the start vertex index of the backrest edge.
        """
        if self.backrest_edge is None:
            return self._find_back_edge()

        if isinstance(self.backrest_edge, int):
            return self.backrest_edge % len(self.base_points)

        # Parse directional specification to identify backrest edge
        direction = self.backrest_edge.lower()
        n = len(self.base_points)
        edge_directions = []

        for i in range(n):
            next_i = (i + 1) % n
            edge_vec = self.base_points[next_i] - self.base_points[i]
            edge_center = (self.base_points[i] + self.base_points[next_i]) / 2
            edge_directions.append({
                'index': i,
                'vector': edge_vec,
                'center': edge_center,
                'length': np.linalg.norm(edge_vec)
            })

        # Map direction string to appropriate edge
        if direction in ['above', 'north', 'top', '+y']:
            return max(edge_directions, key=lambda e: e['center'][1])['index']
        elif direction in ['below', 'south', 'bottom', '-y']:
            return min(edge_directions, key=lambda e: e['center'][1])['index']
        elif direction in ['right', 'east', '+x']:
            return max(edge_directions, key=lambda e: e['center'][0])['index']
        elif direction in ['left', 'west', '-x']:
            return min(edge_directions, key=lambda e: e['center'][0])['index']
        else:
            print(f"Warning: Unknown backrest_edge direction '{self.backrest_edge}', using heuristic")
            return self._find_back_edge()

    def _find_back_edge(self):
        """Fallback heuristic: select edge with smallest average Y coordinate"""
        n = len(self.base_points)
        min_y_avg = float('inf')
        back_edge_idx = 0

        for i in range(n):
            next_i = (i + 1) % n
            y_avg = (self.base_points[i][1] + self.base_points[next_i][1]) / 2
            if y_avg < min_y_avg:
                min_y_avg = y_avg
                back_edge_idx = i

        return back_edge_idx

    def get_vertices_3d(self):
        """Generate 3D vertices with special handling for chair geometry"""
        n_points = len(self.base_points)

        if self.is_chair and self.seat_height is not None:
            # Three-level geometry: floor, seat, backrest
            vertices = np.zeros((n_points * 3, 3))

            # Floor level
            vertices[:n_points, :2] = self.base_points
            vertices[:n_points, 2] = self.z_min

            # Seat level
            vertices[n_points:2*n_points, :2] = self.base_points
            vertices[n_points:2*n_points, 2] = self.seat_height

            # Top level - backrest edge extends to z_max
            back_edge_idx = self._get_backrest_edge_index()
            next_back = (back_edge_idx + 1) % n_points

            vertices[2*n_points:, :2] = self.base_points
            vertices[2*n_points:, 2] = self.seat_height

            # Elevate backrest vertices
            vertices[2*n_points + back_edge_idx, 2] = self.z_max
            vertices[2*n_points + next_back, 2] = self.z_max
        else:
            # Standard box: bottom and top faces
            vertices = np.zeros((n_points * 2, 3))
            vertices[:n_points, :2] = self.base_points
            vertices[:n_points, 2] = self.z_min
            vertices[n_points:, :2] = self.base_points
            vertices[n_points:, 2] = self.z_max

        return vertices

    def get_faces(self):
        """Generate face indices with special handling for chair geometry"""
        n = len(self.base_points)
        faces = []

        if self.is_chair and self.seat_height is not None:
            back_edge_idx = self._get_backrest_edge_index()
            next_back = (back_edge_idx + 1) % n

            # Bottom face and seat surface
            faces.append(list(range(n-1, -1, -1)))
            faces.append(list(range(n, 2*n)))

            # Vertical faces from floor to seat
            for i in range(n):
                next_i = (i + 1) % n
                face = [i, next_i, next_i + n, i + n]
                faces.append(face)

            # Backrest vertical face
            backrest_face = [
                back_edge_idx + n,
                next_back + n,
                next_back + 2*n,
                back_edge_idx + 2*n
            ]
            faces.append(backrest_face)

            # Remaining upper faces
            for i in range(n):
                if i == back_edge_idx:
                    continue
                next_i = (i + 1) % n
                if next_i == next_back:
                    continue
                face = [i + n, next_i + n, next_i + 2*n, i + 2*n]
                faces.append(face)
        else:
            # Standard box faces: bottom, top, and sides
            faces.append(list(range(n-1, -1, -1)))
            faces.append(list(range(n, 2*n)))
            for i in range(n):
                next_i = (i + 1) % n
                face = [i, next_i, next_i + n, i + n]
                faces.append(face)

        return faces

    def to_dict(self):
        """Convert obstacle to dictionary for JSON serialization"""
        return {
            'base_points': self.base_points.tolist(),
            'z_min': self.z_min,
            'z_max': self.z_max,
            'height': self.height,
            'obstacle_type': self.obstacle_type,
            'source_file': self.source_file,
            'seat_height': self.seat_height,
            'backrest_edge': self.backrest_edge
        }

    @classmethod
    def from_dict(cls, data):
        """Create obstacle from dictionary"""
        return cls(
            base_points=data['base_points'],
            z_min=data['z_min'],
            z_max=data['z_max'],
            obstacle_type=data['obstacle_type'],
            source_file=data['source_file'],
            seat_height=data.get('seat_height', None),
            backrest_edge=data.get('backrest_edge', None)
        )


class ObstacleCollection:
    """Collection of obstacles with JSON loading support"""
    def __init__(self):
        self.obstacles = []

    def add_obstacle(self, obstacle):
        """Add an obstacle to the collection"""
        self.obstacles.append(obstacle)

    @classmethod
    def load_json(cls, filepath):
        """Load obstacles from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        collection = cls()
        for obs_data in data['obstacles']:
            collection.add_obstacle(Obstacle.from_dict(obs_data))
        print(f"Loaded {len(collection.obstacles)} obstacles from {filepath}")
        return collection