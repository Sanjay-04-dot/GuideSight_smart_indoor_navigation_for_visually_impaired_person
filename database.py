import sqlite3
import pickle
import numpy as np
from datetime import datetime

class Database:
    def __init__(self, db_path='data/database.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Locations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Frames table - stores visual landmarks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_id INTEGER,
                sequence_number INTEGER,
                image_path TEXT,
                keypoints BLOB,
                descriptors BLOB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (location_id) REFERENCES locations(id)
            )
        ''')
        
        # Navigation graph nodes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_id INTEGER,
                frame_id INTEGER,
                position_index INTEGER,
                FOREIGN KEY (location_id) REFERENCES locations(id),
                FOREIGN KEY (frame_id) REFERENCES frames(id)
            )
        ''')
        
        # Navigation graph edges
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                from_node INTEGER,
                to_node INTEGER,
                distance REAL,
                PRIMARY KEY (from_node, to_node),
                FOREIGN KEY (from_node) REFERENCES graph_nodes(id),
                FOREIGN KEY (to_node) REFERENCES graph_nodes(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_location(self, name):
        """Create new location entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO locations (name) VALUES (?)', (name,))
            location_id = cursor.lastrowid
            conn.commit()
            return location_id
        except sqlite3.IntegrityError:
            # Location already exists
            cursor.execute('SELECT id FROM locations WHERE name = ?', (name,))
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def save_frame(self, location_id, sequence_number, image_path, keypoints, descriptors):
        """Save frame with ORB features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize keypoints (convert to format that can be pickled)
        kp_data = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) 
                   for kp in keypoints]
        kp_blob = pickle.dumps(kp_data)
        desc_blob = pickle.dumps(descriptors)
        
        cursor.execute('''
            INSERT INTO frames (location_id, sequence_number, image_path, keypoints, descriptors)
            VALUES (?, ?, ?, ?, ?)
        ''', (location_id, sequence_number, image_path, kp_blob, desc_blob))
        
        frame_id = cursor.lastrowid
        
        # Create graph node
        cursor.execute('''
            INSERT INTO graph_nodes (location_id, frame_id, position_index)
            VALUES (?, ?, ?)
        ''', (location_id, frame_id, sequence_number))
        
        node_id = cursor.lastrowid
        
        # Create edge from previous node if exists
        if sequence_number > 0:
            cursor.execute('''
                SELECT id FROM graph_nodes 
                WHERE location_id = ? AND position_index = ?
            ''', (location_id, sequence_number - 1))
            prev_node = cursor.fetchone()
            if prev_node:
                # Assume 1.4 m/s walking speed, 0.5 sec interval
                distance = 0.7  # meters
                cursor.execute('''
                    INSERT INTO graph_edges (from_node, to_node, distance)
                    VALUES (?, ?, ?)
                ''', (prev_node[0], node_id, distance))
                # Bidirectional edge
                cursor.execute('''
                    INSERT INTO graph_edges (from_node, to_node, distance)
                    VALUES (?, ?, ?)
                ''', (node_id, prev_node[0], distance))
        
        conn.commit()
        conn.close()
        return frame_id
    
    def get_location_frames(self, location_name):
        """Retrieve all frames for a location"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT f.id, f.sequence_number, f.image_path, f.keypoints, f.descriptors
            FROM frames f
            JOIN locations l ON f.location_id = l.id
            WHERE l.name = ?
            ORDER BY f.sequence_number
        ''', (location_name,))
        
        frames = []
        for row in cursor.fetchall():
            frame_id, seq, img_path, kp_blob, desc_blob = row
            kp_data = pickle.loads(kp_blob)
            # Reconstruct keypoints
            keypoints = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=pt[1], angle=pt[2],
                                      response=pt[3], octave=pt[4], class_id=pt[5]) 
                        for pt in kp_data]
            descriptors = pickle.loads(desc_blob)
            frames.append({
                'id': frame_id,
                'sequence': seq,
                'path': img_path,
                'keypoints': keypoints,
                'descriptors': descriptors
            })
        
        conn.close()
        return frames
    
    def get_all_locations(self):
        """Get list of all saved locations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM locations ORDER BY created_at DESC')
        locations = [row[0] for row in cursor.fetchall()]
        conn.close()
        return locations
    
    def get_navigation_graph(self, location_name):
        """Get complete navigation graph for a location"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all nodes
        cursor.execute('''
            SELECT gn.id, gn.position_index, gn.frame_id
            FROM graph_nodes gn
            JOIN locations l ON gn.location_id = l.id
            WHERE l.name = ?
            ORDER BY gn.position_index
        ''', (location_name,))
        
        nodes = {row[0]: {'position': row[1], 'frame_id': row[2]} 
                for row in cursor.fetchall()}
        
        # Get all edges
        cursor.execute('''
            SELECT ge.from_node, ge.to_node, ge.distance
            FROM graph_edges ge
            JOIN graph_nodes gn ON ge.from_node = gn.id
            JOIN locations l ON gn.location_id = l.id
            WHERE l.name = ?
        ''', (location_name,))
        
        edges = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
        
        conn.close()
        return nodes, edges

import cv2  # Need for KeyPoint reconstruction
