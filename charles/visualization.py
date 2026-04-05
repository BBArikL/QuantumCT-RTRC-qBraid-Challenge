import folium
import random
import math

def get_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def create_route_map(results, coords, depot=0, map_center=None, zoom_start=12):
    if map_center is None:
        map_center = coords[depot]
    
    m = folium.Map(location=map_center, zoom_start=zoom_start)
    
    # depot marker
    folium.Marker(
        location=coords[depot],
        popup="Depot",
        icon=folium.Icon(color="black", icon="home")
    ).add_to(m)
    
    for idx, r in enumerate(results):
        route = r["route"]
        color = get_color()
        
        # ensure route starts/ends at depot
        full_route = [depot] + [n for n in route if n != depot] + [depot]
        route_coords = [coords[n] for n in full_route]
        
        # draw route
        folium.PolyLine(route_coords, color=color, weight=3, opacity=0.8, tooltip=f"Cluster {idx}").add_to(m)
        
        # add customer markers
        for node in r["cluster"]:
            folium.CircleMarker(
                location=coords[node],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Node {node} (Cluster {idx})"
            ).add_to(m)
    
    return m

def save_map(map_object, filename="cvrp_map.html"):
    map_object.save(filename)
    print(f"Map saved to {filename}")


def calculate_total_distance(results, coords, depot=0):
    """
    Calculates the combined distance of all vehicle routes.
    """
    total_dist = 0

    def get_dist(p1, p2):
        # Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2)
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    for r in results:
        route = r.get("route", [])

        # 1. Clean the route and ensure it starts/ends at the depot
        # We filter out any existing depot occurrences to avoid double-counting
        # then wrap the list with the depot index.
        clean_route = [n for n in route if n != depot]
        full_path = [depot] + clean_route + [depot]

        # 2. Sum the segments
        route_dist = 0
        for i in range(len(full_path) - 1):
            start_node = full_path[i]
            end_node = full_path[i + 1]

            p1 = coords[start_node]
            p2 = coords[end_node]

            route_dist += get_dist(p1, p2)

        total_dist += route_dist

    return total_dist