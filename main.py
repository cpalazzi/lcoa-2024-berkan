import pypsa
import p_auxiliary as aux
import glob
import p_data_store as pds
import pandas as pd

"""File to optimise the size of a green ammonia plant given a specified wind and solar profile"""


def generate_network(n_snapshots, data_file):
    """Generates a network that can be used to run several cases"""
    # ==================================================================================================================
    # Set up network
    # ==================================================================================================================

    # Import a generic network
    network = pypsa.Network(override_component_attrs=aux.create_override_components())

    # Set the time values for the network
    network.set_snapshots(range(n_snapshots))

    # Import the design of the H2 plant into the network
    network.import_from_csv_folder(data_file)

    return network


def main(n=None, file_name=None, weather_data=None, multi_site=False, extension=None, aggregation_count=1):
    """Code to execute run at a single location"""
    # Import the weather data
    if file_name is not None and weather_data is None:
        weather_data = aux.get_weather_data(file_name=file_name, aggregation_count=aggregation_count)
    elif file_name is not None and weather_data is not None:
        raise RuntimeWarning('You have entered both a file_name and weather_data; '
                             'the weather_data is being used, and any data in the file will be ignored.')

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
            raise ValueError("You have weather data that isn't in your list of renewables, so I don't know what those "
                             " renewables cost. \n "
                             "Edit generators.csv to include all of the renewables for which you've "
                             " provided data")
    if count != len(renewables_lst):
        raise ValueError("You have renewables for which you haven't provided a weather dataset. "
                         "\n Edit the input file to include all relevant renewables sources, "
                         "or remove the renewables sources from generators.csv")

    # ==================================================================================================================
    # Check if the CAPEX input format in Basic_H2_plant is correct, and fix it up if not
    # ==================================================================================================================

    CAPEX_check = aux.check_CAPEX(file_name=file_name)
    if CAPEX_check is not None:
        for item in [n.generators, n.links, n.stores]:
            item.capital_cost = item.capital_cost * CAPEX_check[1]/100 + item.capital_cost/CAPEX_check[0]

    # ==================================================================================================================
    # Solve the model
    # ==================================================================================================================

    # Ask the user how they would like to solve
    solver, _ = aux.get_solving_info(file_name=file_name)

    # Implement their answer
    n.lopf(solver_name=solver, pyomo=True, extra_functionality=aux.pyomo_constraints)

    # ==================================================================================================================
    # Output results
    # ==================================================================================================================

    if not multi_site:
        # Scale if needed
        scale = aux.get_scale(n, file_name=file_name)

        # Put the results in a nice format
        output = aux.get_results_dict_for_excel(n, scale, aggregation_count=aggregation_count)

        # Send the results to excel
        aux.write_results_to_excel(output, file_name=file_name[12:], extension=extension)
    else:
        output = aux.get_results_dict_for_multi_site(n, aggregation_count=aggregation_count)

    return output


if __name__ == '__main__':
    file_lst = glob.glob('WeatherData/*.csv')
    length = len(pd.read_csv(file_lst[0]))
    network_design = generate_network(length, 'Basic_ammonia_plant_2030')
    store = pds.Data_store()
    for file in glob.glob('Data/*'):
        main(n=network_design, file_name=file, multi_site=True, extension='_2030', aggregation_count=8)
