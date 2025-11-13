import logging
from pcse.base import ParameterProvider

class ParameterDict(dict):
    def add_variable(self, name, value, description=""):
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def copy(self):
        return ParameterDict(self)

def _create_district_specific_parameters(static_row, cropdata):
    """
    V3: Loads pre-calculated physics AND dynamic initial conditions directly
    from the data row.
    This function is now a simple data loader, not a calculator.
    """
    sitedata = ParameterDict()
    soildata = ParameterDict()

    # Site-specific data from the merged dataframe
    sitedata.add_variable('LAT', static_row['latitude'])
    sitedata.add_variable('LON', static_row['longitude'])
    sitedata.add_variable('ELEV', static_row['avg_elevation'])

    # Pass the DYNAMIC, PRE-CALCULATED WAV
    try:
        sitedata.add_variable('WAV', static_row['WAV'])
    except KeyError:
        logging.error(
            f"FATAL: 'WAV' column not found in static_row. The merge with "
            f"initial_conditions_wav.csv failed.")
        raise

    # Add other site-related parameters that are static but not from soil PTFs
    sitedata.add_variable('NOTINF', static_row['NOTINF'])
    sitedata.add_variable('SSMAX', static_row['SSMAX'])

    # Soil-specific, pre-calculated physical data
    soil_params = [
        'SMW', 'SMFCF', 'SM0', 'CRAIRC', 'K0', 'SOPE', 'KSUB', 'RDMSOL']
    for param in soil_params:
        try:
            soildata.add_variable(param, static_row[param])
        except KeyError:
            logging.error(
                f"FATAL: Missing required soil parameter '{param}' in static "
                f"data for district {static_row.get('district_no', 'N/A')}.")
            raise

    # Add remaining model constants/runtime variables required by PCSE
    sitedata.add_variable('IFUNRN', 0.0)
    sitedata.add_variable('SSI', 0.0)
    sitedata.add_variable('SMLIM', soildata['SMFCF'])

    return ParameterProvider(
        cropdata=cropdata, soildata=soildata, sitedata=sitedata), sitedata
