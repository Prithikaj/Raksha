import osmnx as ox
import networkx as nx
import numpy as np

# Load road network for Coimbatore, India
G = ox.graph_from_place("Coimbatore, India", network_type="drive")

crime_zones = set()

def update_crime_zones(crime_locations):
    """
    Updates the global crime_zones set from the reported crime list.
    Also updates edge weights to reflect crime risk.
    """
    global crime_zones
    crime_zones = set(crime_locations)

    for u, v, d in G.edges(data=True):
        # Estimate the midpoint of the edge
        lat = np.mean([G.nodes[u]['y'], G.nodes[v]['y']])
        lon = np.mean([G.nodes[u]['x'], G.nodes[v]['x']])
        # If the midpoint is near any crime zone, increase weight
        if any(np.linalg.norm(np.array([lat, lon]) - np.array(zone)) < 0.002
               for zone in crime_zones):
            d['weight'] = d['length'] * 5  # higher penalty
        else:
            d['weight'] = d['length']  # normal weight

def find_safest_path(start_location, end_location, crime_locations=None):
    """
    Finds the safest path while avoiding high-crime areas if possible.
    Returns (safest_path_coords, alternate_path_coords, caution_message).
    """
    update_crime_zones(crime_locations or [])

    # OSMnx uses (lon, lat) for nearest_nodes, so invert
    orig_node = ox.distance.nearest_nodes(G, start_location[1], start_location[0])
    dest_node = ox.distance.nearest_nodes(G, end_location[1], end_location[0])

    try:
        # Compute the path with 'weight' (crime penalty)
        safest_path = nx.shortest_path(G, orig_node, dest_node, weight='weight')
        path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in safest_path]

        # Check if the path intersects any crime zone
        for (lat, lon) in path_coords:
            if any(np.linalg.norm(np.array([lat, lon]) - np.array(zone)) < 0.002
                   for zone in crime_zones):
                # Attempt to find an alternate route
                alternate_path = find_alternate_path(orig_node, dest_node)
                if alternate_path:
                    return (path_coords,
                            alternate_path,
                            "⚠️ Caution: High-crime area detected! Alternate route suggested.")
                else:
                    return (path_coords,
                            None,
                            "⚠️ No alternate safe route available! Proceed with caution.")

        # If we get here, path is safe
        return (path_coords, None, None)

    except nx.NetworkXNoPath:
        return (None, None, "❌ No route available!")


def find_alternate_path(orig_node, dest_node):
    """
    Finds an alternate path by increasing the penalty even more in crime areas.
    """
    try:
        # Increase penalty further for crime edges
        for u, v, d in G.edges(data=True):
            lat = np.mean([G.nodes[u]['y'], G.nodes[v]['y']])
            lon = np.mean([G.nodes[u]['x'], G.nodes[v]['x']])
            if any(np.linalg.norm(np.array([lat, lon]) - np.array(zone)) < 0.002
                   for zone in crime_zones):
                d['weight'] *= 10

        alt_path = nx.shortest_path(G, orig_node, dest_node, weight='weight')
        return [(G.nodes[node]['y'], G.nodes[node]['x']) for node in alt_path]

    except nx.NetworkXNoPath:
        return None
