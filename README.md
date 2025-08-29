# GPX profile analyzer

**GPX profile analyzer** is an application for analyzing elevation profiles from GPX files.  
It allows you to visualize the route, detect climbs, and generate elevation statistics.  
The project was created as a hobby side-project.

---

## Motivation

The idea behind **GPX profile analyzer** came from my own cycling experience.  
While using Garmin Connect, I often felt that some useful features were missing – especially around climb detection and route profile analysis.  
This project was created to extend those capabilities by providing automatic climb detection, detailed elevation statistics, and an interactive dashboard for route exploration.

---

## Features

- Load route from a GPX file  
- Display the full route map  
- Elevation profile of the entire route or a selected segment  
- Automatic climb detection  
  - customizable minimum length and average slope  
  - climb difficulty classification  
- Table of climbs (distance, elevation gain, average slope, category)  
- Route statistics (uphill / downhill / flat sections)  
- Interactive dashboard built with Streamlit  

---

## Technologies

- Python 3.13.2+  
- Streamlit - UI and dashboard  
- pandas, numpy, geopy - data analysis  
- matplotlib - elevation profile plots  
- folium - route map
- gpxpy - load GPX file  
- black, isort, pre-commit - clean and consistent code  

---

## Installation

1. Clone the repository:  

```bash
git clone https://github.com/MichalMakowiejczuk/GPX-analyzer
cd GPX-analyzer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## Project structure

```bash
climbanalyzer/
│── app.py                  # main app file (Streamlit)
│── components/             # UI modules (sidebar, profile, stats)
│── services/               # application logic
│── scripts/                # GPX processing, profile analysis
│   ├── gpx_parser.py
│   ├── climb_classification.py
│   ├── map_builder.py
│   └── profile/            # facade + supporting classes
│── sample_data/            # example GPX files
│── requirements.txt
│── README.md
```

---

## Demo



---

## Roadmap

- Export results to CSV/Excel
- Descent detection
- Speed calculator

