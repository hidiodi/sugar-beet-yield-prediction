# Sugar Beet Yield Prediction for Germany

This project aims to predict sugar beet yields at the German district level using agronomic, meteorological, and satellite data.

## Setup

1. Clone the repository.
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Run the data processing pipeline:
   `python src/data/03_process_weather_data.py`

## Project Structure
- `data/`: Contains raw, intermediate, and final datasets.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `src/`: Python source code for data processing and modeling.