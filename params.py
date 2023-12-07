'''

    params.py

        Definition of universal model parameters.

    created by:
    - Carlo Palazzi, 2023

'''
# Scaling PyPSA objective for Hydrogen or Ammonia production
# LCOH 
    # output = {
    #     'Headlines': pd.DataFrame({
    #         'Objective function (USD/kg)': [n.objective/(n.loads.p_set.values[0]/39.4*8760*1000)],
    #         'Production (t/year)': n.loads.p_set.values[0]/39.4*8760*scale}, index=['LCOH (USD/kg)']),
    #     'Generators': n.generators.rename(columns={'p_nom_opt': 'Rated Capacity (MW)'})[['Rated Capacity (MW)']]*scale,
    #     'Components': comps,
    #     'Stores': n.stores.rename(columns={'e_nom_opt': 'Storage Capacity (MWh)'})[['Storage Capacity (MWh)']]*scale,
    #     'Energy generation (MW)': n.generators_t.p*scale,
    #     'Energy consumption': consumption,
    #     'Stored energy capacity (MWh)': n.stores_t.e*scale
    # }

# LCOA 
    # output = {
    #     'Headlines': pd.DataFrame({
    #         'Objective function (USD/t)': [float(n.objective) / (n.loads.p_set.values[0] / 6.25 * 8760)],
    #         'Production (t/year)': n.loads.p_set.values[0] / 6.25 * 8760 * scale}, index=['LCOA (USD/t)']),
    #     'Generators': n.generators.rename(columns={'p_nom_opt': 'Rated Capacity (MW)'})[
    #                       ['Rated Capacity (MW)']] * scale,
    #     'Components': comps,
    #     'Stores': scale * aggregation_count * time_step * n.stores.rename(columns={
    #                                     'e_nom_opt': 'Storage Capacity (MWh)'})[['Storage Capacity (MWh)']],
    #     'Energy generation (MW)': n.generators_t.p * scale,
    #     'Energy consumption': consumption,
    #     'Stored energy capacity (MWh)': n.stores_t.e * scale * aggregation_count * time_step
    # }

# Cost assumptions
cost_assumptions = {
    'objective_to_lcoa'         : 1/6.25,
    'objective_to_lcoh'         : 1/39.4,
}

# p_set in loads.py for ammonia is the power required to produce x tonnes of ammonia with HHV=6.25 over 8760 hours
# e.g. for 10Mt p_set = input power = 6.25*10^7/8760 = 7134.7 MW
# For hydrogen should just replace the HHV to 39.4: p_set = 39.4*10^7/8760 = 44977.1689497717

