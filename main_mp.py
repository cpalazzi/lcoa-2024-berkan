import pypsa
import p_auxiliary as aux
import glob
import p_data_store as pds
import pandas as pd
import p_location_class as plc
import geopandas as gpd
from shapely.geometry import Point
import os
import sys
import multiprocessing
import tqdm
from tqdm.contrib.concurrent import process_map
import time


"""File to optimise the size of a green ammonia plant given a specified wind and solar profile"""


# Redirect stdout and stderr to a log file to suppress output.
# This is required when running multiprocessing with terminal outputs in SLURM
# log_file = open("output.log", "w")
# sys.stdout = log_file
# sys.stderr = log_file


def generate_network(n_snapshots, data_file, aggregation_count=1, costs=None, efficiencies=None, time_step=0.5):
    """Generates a network that can be used to run several cases"""
    # ==================================================================================================================
    # Set up network
    # ==================================================================================================================

    # Import a generic network
    network = pypsa.Network(override_component_attrs=aux.create_override_components())

    # Set the time values for the network
    network.set_snapshots(range(int(n_snapshots/aggregation_count)))

    # Import the design of the H2 plant into the network
    network.import_from_csv_folder(data_file)

    if costs is not None:
        for equipment, row in costs.items():
            for df in [network.links, network.generators, network.stores]:
                if equipment in df.index:
                    df.loc[equipment, 'capital_cost'] = row

    if efficiencies is not None:
        for equipment, row in efficiencies.items():
            if equipment in network.links.index:
                network.links.loc[equipment, 'efficiency'] = row

    network.links.loc['HydrogenCompression', 'marginal_cost'] = 0.0001  # Just stops pointless cycling through storage

    water_cost = (39.4/network.links.loc['Electrolysis', 'efficiency'])**(-1) * 10 * 2/0.7  #efficiency/39.4 = t H2/MW
    # 2 USD/kL for water (2/0.7 = AUD); factor of 10 because ~10 t H20/t H2;

    network.links.loc['Electrolysis', 'marginal_cost'] = water_cost

    # Adjust the capital cost of the stores, and the marginal costs, based on the aggregation and number of datapoints
    network.stores.capital_cost *= time_step
    network.links.marginal_cost *= 24*366 / n_snapshots
    # Just in case you have a default that is different to 1 hour in the data
    if aggregation_count is not None:
        network.stores.capital_cost *= aggregation_count
        network.links.marginal_cost *= aggregation_count

    return network


def main(n=None, file_name=None, weather_data=None, multi_site=False, get_complete_output=False,
         extension="", aggregation_count=1, time_step=1.0):
    """Code to execute run at a single location"""
    # Import the weather data
    if file_name is not None and weather_data is None:
        weather_data = aux.get_weather_data(file_name=file_name, aggregation_count=aggregation_count)

    # Import a generic network if needed
    if n is None:
        n = generate_network(len(weather_data), "Basic_ammonia_plant")

    renewables_lst = n.generators.index.to_list()
    # Note: All flows are in MW or MWh, conversions for hydrogen done using HHVs. Hydrogen HHV = 39.4 MWh/t

    # ==================================================================================================================
    # Send the weather data to the model
    # ==================================================================================================================

    count = 0
    for name, dataframe in weather_data.items():
        if name in renewables_lst:
            n.generators_t.p_max_pu[name] = dataframe
            count += 1
        else:
            print(name)
            raise ValueError("You have a columns {a} that isn't in your list of renewables, so I don't know what those "
                             " renewables cost. \n "
                             "Edit generators.csv to include all of the renewables for which you've "
                             " provided data".format(a=name))
    if count != len(renewables_lst):
        raise ValueError("You have renewables for which you haven't provided a weather dataset. "
                         "\n Edit the input file to include all relevant renewables sources, "
                         "or remove the renewables sources from generators.csv")

    # ==================================================================================================================
    # Check if the CAPEX input format in Basic_H2_plant is correct, and fix it up if not
    # ==================================================================================================================

    if not multi_site:
        CAPEX_check = aux.check_CAPEX(file_name=file_name)
        if CAPEX_check is not None:
            for item in [n.generators, n.links, n.stores]:
                item.capital_cost = item.capital_cost * CAPEX_check[1] / 100 + item.capital_cost / CAPEX_check[0]

    # ==================================================================================================================
    # Solve the model
    # ==================================================================================================================

    # Ask the user how they would like to solve, unless they're doing several cases
    if not multi_site:
        solver, _ = aux.get_solving_info(file_name=file_name)
    else:
        solver = 'gurobi'

    # Implement their answer
    n.lopf(solver_name=solver, pyomo=True, extra_functionality=aux.pyomo_constraints)

    # ==================================================================================================================
    # Output results
    # ==================================================================================================================
    if not get_complete_output and not multi_site:
        multi_site = True

    if get_complete_output:
        # Scale if needed
        scale = aux.get_scale(n, file_name=file_name)

        # Put the results in a nice format
        output = aux.get_results_dict_for_excel(n, scale, aggregation_count=aggregation_count, time_step=time_step)

        # Send the results to excel
        aux.write_results_to_excel(output, file_name=file_name[5:], extension=extension)

    if multi_site:
        output = aux.get_results_dict_for_multi_site(n, aggregation_count=aggregation_count, time_step=time_step)

    return output

# Run the code around the world

# Modify the process_location function to accept the 'world' argument, 'data', and 'renewables'
def process_location(lat, lon, world, data, renewables, n):
    # 1. Determine the country for a given latitude and longitude
    print('Processing location')
    intersections = world[world.intersects(Point(lon, lat))]
    if not intersections.empty:
        country = intersections.iloc[0].country
    else:
        country = 'None'

    # 2. Get weather data for that location
    location = plc.renewable_data(data, lat, lon, renewables)

    # 3. Run the optimization code
    result = main(n=n, weather_data=location.concat.drop(columns='Weights'), multi_site=True)
    
    # 4. Return the results along with lat, lon, and country
    return lat, lon, country, result


def run_global(year, data):
    # Download any necessary data
    data_dir = os.path.join(os.getcwd(),'data')
    excel_file = os.path.join(data_dir,'GeneralSteelData.xlsx')
    time_step = 4
    costs = pd.read_excel(excel_file, sheet_name='Costs').set_index('Equipment')[year]
    efficiencies = pd.read_excel(excel_file, sheet_name='Efficiencies').set_index('Equipment')[year]

    # Create a network; override the defaults with the relevant cost and efficiency data
    n = generate_network(8760/time_step, 'Basic_ammonia_plant',
                        costs=costs,
                        efficiencies=efficiencies,
                        aggregation_count=time_step)

    # Get a list of the renewables
    renewables = n.generators.index.to_list()
    print('renewables: '+str(renewables))

    # Create a place to store the data between runs
    store = pds.Data_store()

    # Get the weather data # cpnote: hopefully this pulls in all of the .nc weather data and converts it to the csv input required
    print('data_dir:'+data_dir)
    data = plc.all_locations(data_dir)

    # Get a map of the world to identify which country the weather data is in
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).rename(columns={'name': 'country'})

    # List hte countries you're interested in
    country_lst = world.country
    # country_lst = ['Russia']

    # Parallelize the loop
    num_processes = 4  # Number of parallel processes (adjust according to your server)
    pool = multiprocessing.Pool(processes=num_processes)

    latitudes = list(range(-65, 66))
    min_lon = -170
    max_lon = -160
    longitudes = list(range(min_lon, max_lon))

    print('Beginning starmap')
    results = pool.starmap(process_location, 
                [(lat, lon, world, data, renewables, n) 
                for lat in latitudes for lon in longitudes])

    print('Adding output to the datastore')
    # Add the output to the data store
    for lat, lon, country, result in results:
        store.add_location(lat, lon, country, result)

    # Output all the data at the end
    df = pd.DataFrame.from_dict(store.collated_results, orient='index')
    df.to_csv(f'{year}_lcoa_global_20231105_{min_lon}to{max_lon}mp.csv')

if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(),'data')
    data = plc.all_locations(data_dir)
    
    start_time = time.time()  # Measure the start time
    run_global(2050, data)
    end_time = time.time()  # Measure the end time
    total_time = end_time - start_time
    print(f"Total run time: {total_time} seconds")






