import json
from gmplot import GoogleMapPlotter
import numpy as np
from matplotlib.path import Path
import math
from scipy.optimize import linear_sum_assignment
import random
from itertools import permutations
import heapq
from xml.dom.minidom import getDOMImplementation
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
from xml.dom.minidom import parseString
import zipfile
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Define the common output directory
output_directory = 'output'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

###############################################################################################
#####################             0 Generator Area Preprocess              ####################
###############################################################################################
def validate_json_structure(data):
    if "mission" not in data:
        return False
    if "name" not in data["mission"]:
        return False
    if "waypoints" not in data["mission"]:
        return False
    for point in data["mission"]["waypoints"]:
        if "latitude" not in point or "longitude" not in point:
            return False
    return True

def preprocess_and_generate_json(data):
    try:
        is_valid = validate_json_structure(data)
        if is_valid:
            initial_points = data['mission']['waypoints']
            # Ensure the route is closed by adding a landing point if necessary
            if (initial_points[0]['latitude'] != initial_points[-1]['latitude']) or (initial_points[0]['longitude'] != initial_points[-1]['longitude']):
                land_point = {'latitude': initial_points[0]['latitude'], 'longitude': initial_points[0]['longitude']}
                land_point['id'] = len(initial_points) + 1  
                initial_points.append(land_point) 
            # Reorder and update waypoints
            for i, point in enumerate(initial_points):
                reordered_point = {'id': i + 1}
                reordered_point['latitude'] = point['latitude']
                reordered_point['longitude'] = point['longitude']
                initial_points[i] = reordered_point 
            # Set actions for the waypoints
            initial_points[0]['action'] = 'takeoff'  
            initial_points[-1]['action'] = 'land'  
            for waypoint in initial_points[1:-1]:
                waypoint['action'] = 'waypoint'
            # Write to output file
            path = os.path.join(output_directory, 'generated_area.json')
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)
            print("\033[92mThe area points have been updated and saved to:\033[0m", path)
        else:
                print("\033[91mThe area JSON structure is invalid \033[0m ")
    except FileNotFoundError:
        print("The specified file was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON from the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def generate_visualization_map():
    path = os.path.join(output_directory, 'generated_area.json')
    with open(path, 'r') as file:
        data = json.load(file)
    waypoints = data['mission']['waypoints']
    latitudes = [wp['latitude'] for wp in waypoints]
    longitudes = [wp['longitude'] for wp in waypoints]
    center_latitude = sum(latitudes) / len(latitudes)
    center_longitude = sum(longitudes) / len(longitudes)
    gmap = GoogleMapPlotter(center_latitude, center_longitude, 17, map_type='satellite')
    for wp in waypoints:
        action = wp['action']
        if action == 'takeoff':
            gmap.marker(wp['latitude'], wp['longitude'], 'yellow', title="Takeoff")
        elif action == 'land':
            gmap.marker(wp['latitude'], wp['longitude'], 'yellow', title="Land")
        else: 
            gmap.marker(wp['latitude'], wp['longitude'], 'red', title="Waypoint")
    path_waypoints = [wp for wp in waypoints if wp['action'] in ["takeoff", "land", "waypoint"]]
    polygon_lats = [wp['latitude'] for wp in path_waypoints]
    polygon_lons = [wp['longitude'] for wp in path_waypoints]
    gmap.plot(polygon_lats, polygon_lons, 'red', edge_width=2.5)
    output_html_path = os.path.join(output_directory, 'generated_area_map.html')
    gmap.draw(output_html_path)
    print(f"\033[92mMap visualization saved to:\033[0m {output_html_path}")
    print("\n")




###############################################################################################
##########################             1 Generator Grid              ##########################
###############################################################################################
def generate_grid():
    path1 = os.path.join(output_directory, 'generated_area.json')
    path2 = os.path.join(output_directory, 'generated_grid.json')
    grid_spacing = 0.0006
    with open(path1, 'r') as file:
        data = json.load(file)
    waypoints = data['mission']['waypoints']
    last_action = waypoints[-1]['action'] if waypoints else None
    if last_action == 'land':
        land_waypoint = waypoints.pop()
    for wp in waypoints:
        if wp['action'] not in ['takeoff', 'land']:
            wp['action'] = 'polygon waypoint'
    poly_path = Path([(wp['longitude'], wp['latitude']) for wp in waypoints if wp['action'] == 'polygon waypoint'])
    latitudes = [wp['latitude'] for wp in waypoints]
    longitudes = [wp['longitude'] for wp in waypoints]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    grid_lat = np.arange(min_lat, max_lat, grid_spacing)
    grid_lon = np.arange(min_lon, max_lon, grid_spacing)
    grid_points = np.meshgrid(grid_lon, grid_lat)
    grid_points = np.vstack([grid_points[0].ravel(), grid_points[1].ravel()]).T
    inside_points = poly_path.contains_points(grid_points)
    grid_waypoints = [{'latitude': lat, 'longitude': lon, 'action': 'waypoint'} 
                    for lon, lat in grid_points[inside_points]]
    combined_waypoints = waypoints + grid_waypoints
    if last_action == 'land':
        combined_waypoints.append(land_waypoint)
    for i, waypoint in enumerate(combined_waypoints, start=1):
        waypoint['id'] = i
    data['mission']['waypoints'] = combined_waypoints
    with open(path2, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"\033[92mGrid waypoints have been saved to:\033[0m {path2}")


def generate_grid_map():
    path = os.path.join(output_directory, 'generated_grid.json')
    with open(path, 'r') as file:
        combined_data = json.load(file)
    combined_waypoints = combined_data['mission']['waypoints']
    latitudes = [wp['latitude'] for wp in combined_waypoints]
    longitudes = [wp['longitude'] for wp in combined_waypoints]
    center_latitude = sum(latitudes) / len(latitudes)
    center_longitude = sum(longitudes) / len(longitudes)
    gmap = GoogleMapPlotter(center_latitude, center_longitude, 17, map_type='satellite')
    for wp in combined_waypoints:
        action = wp['action']
        if action == 'takeoff':
            gmap.marker(wp['latitude'], wp['longitude'], 'yellow', title="Takeoff")
        elif action == 'land':
            gmap.marker(wp['latitude'], wp['longitude'], 'yellow', title="Land")
        elif action == 'polygon waypoint':
            gmap.marker(wp['latitude'], wp['longitude'], 'red', title="Polygon Waypoint")
        else: 
            gmap.marker(wp['latitude'], wp['longitude'], 'orange', title="Waypoint")
    path_waypoints = [wp for wp in combined_waypoints if wp['action'] in ["takeoff", "land", "polygon waypoint"]]
    polygon_lats = [wp['latitude'] for wp in path_waypoints]
    polygon_lons = [wp['longitude'] for wp in path_waypoints]
    gmap.plot(polygon_lats, polygon_lons, 'red', edge_width=2.5)
    path = os.path.join(output_directory, 'generated_grid_map.html')
    gmap.draw(path)
    print(f"\033[92mMap visualization saved to:\033[0m {path}")




###############################################################################################
##############             2 Optimization Shortest Path Algorithms              ###############
###############################################################################################
def calculate_distance(wp1, wp2):
    R = 6373000.0  # Approximate radius of Earth in meters
    lat1 = math.radians(wp1['latitude'])
    lon1 = math.radians(wp1['longitude'])
    lat2 = math.radians(wp2['latitude'])
    lon2 = math.radians(wp2['longitude'])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def random_algorithm(waypoints):
    print("\nOptimization Algorithm: \033[92mRandom\033[0m")
    if not waypoints:
        return [], 0
    path = [waypoints[0]] # Start from the first waypoint
    total_distance = 0
    shuffled_points = waypoints[1:-1] # Exclude the first and last points for shuffling
    random.shuffle(shuffled_points)
    path.extend(shuffled_points)
    path.append(waypoints[-1]) # Ensure the path ends with the last waypoint to complete the cycle
    for i in range(len(path) - 1):
        total_distance += calculate_distance(path[i], path[i + 1])
    total_distance += calculate_distance(path[-1], path[0]) # Add distance from last to first to complete the cycle
    return path, total_distance

def perimetric_sequential_algorithm(waypoints):
    print("\nAlgorithm: \033[92mPerimetric Sequential\033[0m")
    filtered_waypoints = [wp for wp in waypoints if wp['action'] in ["initial waypoint", "takeoff", "land"]]
    filtered_sequential_path = filtered_waypoints[:] # Copy the filtered list
    total_distance = 0.0
    for i in range(1, len(filtered_sequential_path)):
        total_distance += calculate_distance(filtered_sequential_path[i-1], filtered_sequential_path[i])
    return filtered_sequential_path, total_distance

def perimetric_sequential_algorithm(waypoints):
    print("\nAlgorithm: \033[92mPerimetric Sequential\033[0m")
    # Filter waypoints that are part of the initial polygon
    polygon_waypoints = [wp for wp in waypoints if wp['action'] == 'polygon waypoint']
    if 'takeoff' in [wp['action'] for wp in waypoints]:
        polygon_waypoints.insert(0, next(wp for wp in waypoints if wp['action'] == 'takeoff'))
    if 'land' in [wp['action'] for wp in waypoints]:
        polygon_waypoints.append(next(wp for wp in waypoints if wp['action'] == 'land'))
    # Calculate the total distance for the filtered sequential path
    total_distance = 0.0
    for i in range(1, len(polygon_waypoints)):
        total_distance += calculate_distance(polygon_waypoints[i-1], polygon_waypoints[i])
    total_distance += calculate_distance(polygon_waypoints[-1], polygon_waypoints[0])
    
    return polygon_waypoints, total_distance

def nearest_neighbor(waypoints):
    print("\nAlgorithm: \033[92mNearest Neighbor\033[0m")
    start = waypoints.pop(0)
    end = waypoints.pop(-1)
    optimized_path = [start]
    total_distance = 0.0
    while waypoints:
        nearest = min(waypoints, key=lambda wp: calculate_distance(optimized_path[-1], wp))
        total_distance += calculate_distance(optimized_path[-1], nearest)
        optimized_path.append(nearest)
        waypoints.remove(nearest)
    total_distance += calculate_distance(optimized_path[-1], end)
    optimized_path.append(end)
    return optimized_path, total_distance

def hungarian_algorithm(waypoints):
    print("\nAlgorithm: \033[92mHungarian\033[0m")
    num_waypoints = len(waypoints)
    distance_matrix = np.zeros((num_waypoints, num_waypoints))
    for i in range(num_waypoints):
        for j in range(num_waypoints):
            if i != j:
                distance_matrix[i, j] = calculate_distance(waypoints[i], waypoints[j])
            else:
                distance_matrix[i, j] = np.inf
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    path = [waypoints[i] for i in row_ind]
    total_distance = sum(calculate_distance(path[i], path[(i + 1) % num_waypoints]) for i in range(num_waypoints))
    return path, total_distance

def brute_force_algorithm(waypoints):
    print("\nAlgorithm: \033[92mBrute Force\033[0m")
    if len(waypoints) > 3:
        print("\033[91mError: Too many waypoints to run Brute Force algorithm.\033[0m")
        return None, None
    start = waypoints[0]
    end = waypoints[-1]
    middle_waypoints = waypoints[1:-1]
    shortest_path = None
    min_distance = float('inf')
    for perm in permutations(middle_waypoints):
        current_path = [start] + list(perm) + [end]
        current_distance = 0
        for i in range(len(current_path) - 1):
            current_distance += calculate_distance(current_path[i], current_path[i + 1])
        if current_distance < min_distance:
            min_distance = current_distance
            shortest_path = current_path
    return shortest_path, min_distance


def export_to_kml(waypoints, filename):
    # Create the root element
    kml = Element('kml', xmlns="http://www.opengis.net/kml/2.2", **{'xmlns:wpml': "http://www.dji.com/wpmz/1.0.0"})
    # Create the Document element
    document = SubElement(kml, 'Document')
    # Create the wpml:missionConfig element
    mission_config = SubElement(document, 'wpml:missionConfig')
    SubElement(mission_config, 'wpml:flyToWaylineMode').text = 'safely'
    SubElement(mission_config, 'wpml:finishAction').text = 'goHome'
    SubElement(mission_config, 'wpml:exitOnRCLost').text = 'executeLostAction'
    SubElement(mission_config, 'wpml:executeRCLostAction').text = 'goBack'
    SubElement(mission_config, 'wpml:takeOffSecurityHeight').text = '0'
    SubElement(mission_config, 'wpml:globalTransitionalSpeed').text = '5'
    # Create the wpml:droneInfo element
    drone_info = SubElement(mission_config, 'wpml:droneInfo')
    SubElement(drone_info, 'wpml:droneEnumValue').text = '0'
    SubElement(drone_info, 'wpml:droneSubEnumValue').text = '0'
    # Create the wpml:payloadInfo element
    payload_info = SubElement(mission_config, 'wpml:payloadInfo')
    SubElement(payload_info, 'wpml:payloadEnumValue').text = '0'
    SubElement(payload_info, 'wpml:payloadSubEnumValue').text = '0'
    SubElement(payload_info, 'wpml:payloadPositionIndex').text = '0'
    # Create the Folder element
    folder = SubElement(document, 'Folder')
    SubElement(folder, 'wpml:templateType').text = 'waypoint'
    SubElement(folder, 'wpml:templateId').text = '0'
    # Create the wpml:waylineCoordinateSysParam element
    wayline_coord_sys_param = SubElement(folder, 'wpml:waylineCoordinateSysParam')
    SubElement(wayline_coord_sys_param, 'wpml:coordinateMode').text = 'WGS84'
    SubElement(wayline_coord_sys_param, 'wpml:heightMode').text = 'relativeToStartPoint'
    SubElement(wayline_coord_sys_param, 'wpml:positioningType').text = 'GPS'
    # Add global settings
    SubElement(folder, 'wpml:autoFlightSpeed').text = '4.056'
    SubElement(folder, 'wpml:globalHeight').text = '30.000'
    SubElement(folder, 'wpml:caliFlightEnable').text = '0'
    SubElement(folder, 'wpml:gimbalPitchMode').text = 'usePointSetting'
    # Create the wpml:globalWaypointHeadingParam element
    global_waypoint_heading_param = SubElement(folder, 'wpml:globalWaypointHeadingParam')
    SubElement(global_waypoint_heading_param, 'wpml:waypointHeadingMode').text = 'followWayline'
    SubElement(global_waypoint_heading_param, 'wpml:waypointHeadingAngle').text = '0'
    SubElement(global_waypoint_heading_param, 'wpml:waypointPoiPoint').text = '0.000000,0.000000,0.000000'

    SubElement(folder, 'wpml:globalWaypointTurnMode').text = 'toPointAndStopWithDiscontinuityCurvature'
    SubElement(folder, 'wpml:globalUseStraightLine').text = '0'

    # Add the placemarks for each waypoint
    for index, wp in enumerate(waypoints):
        placemark = SubElement(folder, 'Placemark')
        point = SubElement(placemark, 'Point')
        coordinates = SubElement(point, 'coordinates')
        coordinates.text = f"{wp['longitude']},{wp['latitude']},0"
        
        SubElement(placemark, 'wpml:index').text = str(index)
        SubElement(placemark, 'wpml:ellipsoidHeight').text = '30.000'  # Example value, adjust as needed
        SubElement(placemark, 'wpml:height').text = '30.000'  # Example value, adjust as needed
        SubElement(placemark, 'wpml:useGlobalHeight').text = '0'
        SubElement(placemark, 'wpml:useGlobalSpeed').text = '1'
        SubElement(placemark, 'wpml:useGlobalHeadingParam').text = '1'
        SubElement(placemark, 'wpml:useGlobalTurnParam').text = '1'
        SubElement(placemark, 'wpml:gimbalPitchAngle').text = '0'  # Example value, adjust as needed
        SubElement(placemark, 'wpml:useStraightLine').text = '0'

    # Convert to string and prettify the XML
    rough_string = tostring(kml, 'utf-8')
    reparsed = parseString(rough_string)
    pretty_string = reparsed.toprettyxml(indent="  ")
    # Save to file
    with open(filename, "w") as f:
        f.write(pretty_string)


def convert_kml_to_kmz(kml_file_name, kmz_file_name=None):
    if kmz_file_name is None:
        kmz_file_name = os.path.splitext(kml_file_name)[0] + '.kmz'
    with zipfile.ZipFile(kmz_file_name, 'w', zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr('wpmz/', '')
        kmz.write(kml_file_name, 'wpmz/template.kml')


def generate_optimization():
    path = os.path.join(output_directory, 'generated_grid.json')
    with open(path, 'r') as file:
        data = json.load(file)
    waypoints = data['mission']['waypoints']
    optimization_option = data['mission']['name']

    if optimization_option == 'Randomized Patrol':
        optimized_path, total_distance = random_algorithm(waypoints.copy())
    elif optimization_option == "Perimetric Patrol":
        optimized_path, total_distance = perimetric_sequential_algorithm(waypoints.copy())
    elif optimization_option == "Optimization V1":
        optimized_path, total_distance = nearest_neighbor(waypoints.copy())
    elif optimization_option == "Optimization V2":
        optimized_path, total_distance = hungarian_algorithm(waypoints.copy())
    elif optimization_option == "Emergency":
        optimized_path, total_distance = brute_force_algorithm(waypoints.copy())
    else:
        print("\033[91mInvalid optimization option from user\033[0m")
        return jsonify({"Error": "Invalid algorithm name"}), 400


    if ((optimized_path != None) and  (total_distance != None)):
        export_json_path = os.path.join(output_directory, 'generated_optimized_path.json')
        data = {"mission": {"name": "Optimized Mission Path", "waypoints": optimized_path}}
        with open(export_json_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)
        print(f"\033[92mOptimized mission path exported to:\033[0m {export_json_path}")
        print(f"Total distance: \033[92m{total_distance:0.0f} meters\033[0m")

        latitudes = [wp['latitude'] for wp in optimized_path]
        longitudes = [wp['longitude'] for wp in optimized_path]
        gmap = GoogleMapPlotter(np.mean(latitudes), np.mean(longitudes), 16, map_type='satellite')
        for wp in optimized_path:
            gmap.marker(wp['latitude'], wp['longitude'], 'orange')
            action = wp['action']
            if action in ["polygon waypoint"]:
                gmap.marker(wp['latitude'], wp['longitude'], 'red', title=f"{action} (ID: {wp['id']})")
            if action in ["takeoff", "land"]:
                gmap.marker(wp['latitude'], wp['longitude'], 'yellow', title=f"{action} (ID: {wp['id']})")
        gmap.plot(latitudes, longitudes, 'yellow', edge_width=2.5)
        output_html = os.path.join(output_directory, "generated_wayline_optimized_path_map.html")
        gmap.draw(output_html)
        print(f"\033[92mMap visualization exported to:\033[0m {output_html}")
        print("\n")

        export_kml_path = os.path.join(output_directory, "generated_wayline_optimized_plan.kml")
        export_to_kml(optimized_path, export_kml_path)
        print(f"\033[92mOptimized KML flight path exported to:\033[0m {export_kml_path}")

        export_kmz_path = os.path.join(output_directory, 'generated_wayline_optimized_plan.kmz')
        convert_kml_to_kmz(export_kml_path, export_kmz_path)
        print(f"\033[92mOptimized KMZ flight path exported to:\033[0m {export_kmz_path}")
        print("\n")

    else:
        print("\033[91mError: Optimization algorithm returned None \033[0m ")



###############################################################################################
#####################                     Listener APP                     ####################
############################################################################################### 
app = Flask(__name__)
CORS(app)

@app.route('/api/import', methods=['POST'])
def receive_json():
    if request.is_json:
        data = request.get_json()
        print("\033[92mReceived JSON data:\033[0m")
        print(json.dumps(data, indent=4))
        preprocess_and_generate_json(data)
        generate_visualization_map()
        generate_grid()
        generate_grid_map()
        generate_optimization()
        return jsonify({"message": "JSON received and processed"}), 200
    else:
        return jsonify({"error": "Invalid JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)








