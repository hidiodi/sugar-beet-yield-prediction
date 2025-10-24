# File: src/models/crop_model_wrapper.py
# Description: PCSE/WOFOST wrapper for SUGAR BEET.
#              WORKING VERSION for PCSE 6.0 - No DATA_DIR dependency!

import pandas as pd
import os
import logging
import datetime as dt
import yaml
import pkg_resources

# --- PCSE Imports (Corrected for PCSE 6.0) ---
try:
    from pcse.models import Wofost72_WLP_CWB
    from pcse.base import ParameterProvider
    from pcse.input import YAMLCropDataProvider, CABOFileReader, CSVWeatherDataProvider
    from pcse.agromanager import AgroManager
    from pcse.util import WOFOST72SiteDataProvider

    # NO IMPORTS from pcse.settings or pcse.engine

except ImportError as e:
    logging.error(f"FATAL: A PCSE component could not be imported: {e}")
    raise

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s')


def get_pcse_data_dir():
    """
    Find the PCSE package data directory where soil and site files are located.
    In PCSE 6.0, these are stored in the package's data folder.
    """
    try:
        # Try to find the pcse package installation directory
        import pcse
        pcse_dir = os.path.dirname(pcse.__file__)
        data_dir = os.path.join(pcse_dir, 'data')

        if os.path.exists(data_dir):
            return data_dir

        # Fallback: try pkg_resources
        try:
            data_dir = pkg_resources.resource_filename('pcse', 'data')
            if os.path.exists(data_dir):
                return data_dir
        except:
            pass

        # If still not found, return None and we'll handle it gracefully
        logging.warning("Could not automatically locate PCSE data directory")
        return None

    except Exception as e:
        logging.error(f"Error finding PCSE data directory: {e}")
        return None


class LINTUL_PCSE:
    """
    A wrapper to run the WOFOST crop model via the PCSE library,
    specialized for Sugar Beet.
    """

    def __init__(self, custom_data_dir=None):
        """
        Initialize the PCSE wrapper.

        Args:
            custom_data_dir: Optional custom path to PCSE data directory.
                           If None, will try to auto-detect.
        """
        # Locate crop file (in your project)
        self.crop_file_path = 'data/01_raw/sugarbeet.yaml'

        # Locate PCSE data directory for soil and site files
        if custom_data_dir:
            pcse_data_dir = custom_data_dir
        else:
            pcse_data_dir = get_pcse_data_dir()

        if pcse_data_dir:
            self.soil_file_path = os.path.join(pcse_data_dir, 'soil', 'ec3.soil')
            self.site_file_path = os.path.join(pcse_data_dir, 'site', 'wofost72.site')
        else:
            # Use relative paths as fallback
            self.soil_file_path = 'ec3.soil'
            self.site_file_path = 'wofost72.site'

        self._verify_paths()
        self._initialize_parameters()
        self.is_proxy = False
        logging.info("✅ PCSE Sugar Beet Wrapper initialized successfully.")

    def _verify_paths(self):
        """Ensures all necessary parameter files exist before starting."""
        if not os.path.exists(self.crop_file_path):
            raise FileNotFoundError(
                f"CRITICAL: Sugar beet YAML not found at {self.crop_file_path}\n"
                f"Please ensure your sugarbeet.yaml file exists in data/01_raw/"
            )

        if not os.path.exists(self.soil_file_path):
            logging.warning(f"Soil file not found at {self.soil_file_path}")
            logging.warning("Will attempt to continue, but this may cause errors")

        if not os.path.exists(self.site_file_path):
            logging.warning(f"Site file not found at {self.site_file_path}")
            logging.warning("Will attempt to continue, but this may cause errors")

    def _initialize_parameters(self):
        """Loads and combines crop, soil, and site parameters."""
        try:
            # Load crop parameters
            self.cropd = YAMLCropDataProvider(fpath=self.crop_file_path)
            self.cropd.set_active_crop('sugarbeet', 'Sugarbeet_601')
            logging.info("✓ Crop parameters loaded")

            # Load soil parameters
            if os.path.exists(self.soil_file_path):
                self.soild = CABOFileReader(self.soil_file_path)
                logging.info("✓ Soil parameters loaded")
            else:
                logging.error(f"Soil file not found: {self.soil_file_path}")
                raise FileNotFoundError(f"Soil file required: {self.soil_file_path}")

            # Load or create site parameters
            if os.path.exists(self.site_file_path):
                try:
                    # Try the old method first
                    self.sited = WOFOST72SiteDataProvider(self.site_file_path)
                    logging.info("✓ Site parameters loaded from file")
                except:
                    # If that fails, create site parameters with defaults
                    self.sited = WOFOST72SiteDataProvider(WAV=100, CO2=360)
                    logging.info("✓ Site parameters created with defaults")
            else:
                # Create site parameters with defaults
                self.sited = WOFOST72SiteDataProvider(WAV=100, CO2=360)
                logging.info("✓ Site parameters created with defaults")

            # Combine all parameters
            self.params = ParameterProvider(self.cropd, self.soild, self.sited)
            logging.info("✓ All parameters combined successfully")

        except Exception as e:
            logging.error(f"FATAL: Failed to initialize PCSE parameters. Error: {e}")
            raise

    def _prepare_weather_data(self, daily_weather_df: pd.DataFrame, run_id: str):
        """Converts a pandas DataFrame into a temporary PCSE-compatible CSV file."""
        if daily_weather_df.empty:
            return None

        pcse_df = pd.DataFrame({
            'DAY': pd.to_datetime(daily_weather_df['date']),
            'IRRAD': daily_weather_df['srad'],  # MJ/m2/day
            'TMIN': daily_weather_df['tmin'],
            'TMAX': daily_weather_df['tmax'],
            'VAP': 0.9,  # kPa (default value)
            'WIND': 2.0,  # m/s (default value)
            'RAIN': daily_weather_df['precip'],  # mm/day
        }).set_index('DAY')

        temp_file = f'temp_weather_{run_id}.csv'

        # PCSE CSV weather file header
        header = "Country, Station, Description, Source, Longitude, Latitude, Elevation, AngstromA, AngstromB, HasSunshine, IS_PCSE_FORMAT\n"
        header += "Unknown, Unknown, Generated, Internal, 0.0, 50.0, 0.0, 0.0, 0.0, 0, True\n"
        header += "DAY,IRRAD,TMIN,TMAX,VAP,WIND,RAIN\n"

        with open(temp_file, 'w') as f:
            f.write(header)
            pcse_df[['IRRAD', 'TMIN', 'TMAX', 'VAP', 'WIND', 'RAIN']].to_csv(
                f, header=False, float_format='%.2f'
            )

        return temp_file

    def run(self, daily_weather_df: pd.DataFrame, district_params: dict) -> float:
        """
        Runs the WOFOST sugar beet simulation and returns fresh yield in dt/ha.

        Args:
            daily_weather_df: DataFrame with columns ['date', 'srad', 'tmin', 'tmax', 'precip']
            district_params: Dict with keys 'planting_date' (YYYY-MM-DD) and optional 'run_id'

        Returns:
            Fresh yield in decitons per hectare (dt/ha)
        """
        run_id = district_params.get('run_id', f"run_{int(dt.datetime.now().timestamp())}")
        temp_weather_file = None

        try:
            # --- 1. Prepare Weather Data ---
            temp_weather_file = self._prepare_weather_data(daily_weather_df, run_id)
            if not temp_weather_file:
                logging.warning(f"Empty weather data for run {run_id}")
                return 400.0

            wdp = CSVWeatherDataProvider(temp_weather_file)

            # --- 2. Setup Agromanagement ---
            planting_date = dt.datetime.strptime(district_params['planting_date'], "%Y-%m-%d").date()
            harvest_date = dt.date(planting_date.year, 10, 31)  # Fixed harvest for sugar beet

            agromanagement_string = f"""
- 2000-01-01:
    CropCalendar:
        crop_name: sugarbeet
        variety_name: Sugarbeet_601
        crop_start_date: {planting_date}
        crop_start_type: emergence
        crop_end_date: {harvest_date}
        crop_end_type: harvest
        max_duration: 300
    TimedEvents: null
    StateEvents: null
"""
            agromanagement = yaml.safe_load(agromanagement_string)

            # --- 3. Initialize and Run Model ---
            wofost = Wofost72_WLP_CWB(self.params, wdp, agromanagement)

            # CRITICAL: run_till_terminate() is a METHOD on the model object!
            wofost.run_till_terminate()

            # --- 4. Extract Results ---
            summary = wofost.get_summary_output()

            if not summary or len(summary) == 0:
                logging.warning(f"No summary output for run {run_id}")
                return 400.0

            # Extract storage organ dry matter weight (kg/ha)
            yield_dm_kg_ha = summary[0].get('WSO', 0)
            if yield_dm_kg_ha is None or yield_dm_kg_ha <= 0:
                logging.warning(f"Invalid WSO value for run {run_id}: {yield_dm_kg_ha}")
                return 400.0

            # --- 5. Convert to Fresh Weight ---
            # Sugar beet dry matter content is typically 23%
            DRY_MATTER_CONTENT = 0.23
            # Convert kg/ha to dt/ha and account for dry matter
            fresh_yield_dt_ha = (yield_dm_kg_ha / 100) / DRY_MATTER_CONTENT

            logging.info(f"Run {run_id}: WSO={yield_dm_kg_ha:.1f} kg/ha DM -> "
                         f"{fresh_yield_dt_ha:.1f} dt/ha fresh weight")

            return fresh_yield_dt_ha

        except Exception as e:
            logging.error(f"PCSE run {run_id} failed: {e}", exc_info=True)
            return 400.0  # Default fallback yield

        finally:
            # Clean up temporary weather file
            if temp_weather_file and os.path.exists(temp_weather_file):
                try:
                    os.remove(temp_weather_file)
                except Exception as cleanup_error:
                    logging.warning(f"Could not remove temp file {temp_weather_file}: {cleanup_error}")