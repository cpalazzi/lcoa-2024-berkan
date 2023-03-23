"""Creates a class in which the optimisation driver stores results"""
class Data_store:
    """Class designed to store data from each instance of the optimisation driver case"""
    def __init__(self):
        """Creates an empty dictionary in which data will be stored"""
        self.collated_results = {}

    def add_location(self, lat, lon, country, location_results):
        """Adds a location to the collated results"""
        dct = {'Latitude': lat, 'Longitude': lon, 'Country': country}
        for key in location_results.keys():
            dct[key] = location_results[key]
        self.collated_results['{a}_{b}'.format(a=lat, b=lon)] = dct
                