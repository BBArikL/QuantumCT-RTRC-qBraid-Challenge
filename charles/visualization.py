import folium
import random

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