# Traffic Accident Analysis | Task-05

**Analyze traffic accident data to identify patterns related to road conditions, weather, and time of day. Visualize accident hotspots and contributing factors.**

## 📋 Project Overview

This project performs comprehensive analysis of US traffic accident data to uncover patterns and insights that can help improve road safety. The analysis explores temporal patterns, weather conditions, geographic hotspots, and contributing factors to accidents.

## 🎯 Objectives

- Analyze accident patterns by time of day, day of week, and season
- Identify weather conditions associated with higher accident rates
- Visualize geographic hotspots where accidents frequently occur
- Examine contributing factors and their correlations with accident severity
- Generate actionable insights for traffic safety improvements

## 📊 Dataset

**Source:** [US Accidents Dataset (Kaggle)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

The dataset contains information about traffic accidents across the United States, including:
- Location (latitude, longitude)
- Time and date
- Weather conditions
- Road conditions
- Severity levels
- Environmental factors

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualizations

## 📦 Installation

1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the project directory

## 🚀 Usage

Run the analysis script:
```bash
python traffic_accident_analysis.py
```

The script will:
1. Load and preprocess the accident data
2. Extract time-based features
3. Perform exploratory data analysis
4. Generate visualizations
5. Create a comprehensive summary report

## 📈 Output Files

The analysis generates the following files:

### Visualizations
- `accidents_by_time.png` - Hourly and daily accident patterns
- `accidents_by_time_period.png` - Distribution across time periods
- `weather_conditions.png` - Weather impact analysis
- `severity_distribution.png` - Accident severity breakdown
- `geographic_hotspots.png` - Geographic distribution heatmap
- `correlation_matrix.png` - Environmental factors correlation

### Report
- `analysis_summary.txt` - Detailed findings and recommendations

## 🔍 Key Findings

The analysis reveals:
- **Peak accident times**: Rush hours (7-9 AM, 4-6 PM) show significantly higher rates
- **Day patterns**: Weekdays experience more accidents than weekends
- **Weather impact**: Certain weather conditions correlate with increased accident frequency
- **Geographic patterns**: Accidents cluster in major urban areas and specific highway segments
- **Severity factors**: Environmental conditions influence accident severity

## 💡 Recommendations

Based on the analysis:
1. Increase traffic enforcement during identified peak hours
2. Implement weather-responsive traffic management systems
3. Focus safety campaigns on high-risk time periods and locations
4. Deploy additional resources to geographic hotspots
5. Improve road conditions in areas prone to severe accidents

## 📝 Project Structure

```
.
├── traffic_accident_analysis.py    # Main analysis script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── output/                         # Generated visualizations and reports
```

## 🎓 Skills Demonstrated

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Time series analysis
- Geospatial data visualization
- Statistical correlation analysis
- Data-driven insights generation
- Professional reporting

## 👤 Author

**Partey45**

## 📄 License

This project is part of the Prodigy InfoTech internship program.

## 🙏 Acknowledgments

- Dataset provided by [Sobhan Moosavi](https://www.kaggle.com/sobhanmoosavi)
- Prodigy InfoTech for the internship opportunity

---

*Completed as part of Task-05 for Prodigy InfoTech Data Science Internship*
