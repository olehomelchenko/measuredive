import pandas as pd
import streamlit as st
import altair as alt

import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


def create_anomaly_chart(data, date_column, metric_column, anomalies, chart_title):
    # Base chart with main metric
    base = alt.Chart(data.reset_index()).encode(
        x=alt.X(f'{date_column}:T', title='Date'),
        y=alt.Y(f'{metric_column}:Q', title=metric_column)
    )
    
    # Line chart for the metric
    line = base.mark_line(color='steelblue').properties(
        width=800,
        height=400,
        title=chart_title
    )
    
    # Add points for spikes
    spikes = alt.Chart(anomalies['spikes']).mark_point(
        color='red',
        size=100
    ).encode(
        x='date:T',
        y='value:Q',
        tooltip=['date:T', 'value:Q', 'z_score:Q']
    )
    
    # Add points for level shifts
    shifts = alt.Chart(anomalies['level_shifts']).mark_point(
        color='orange',
        size=100
    ).encode(
        x='date:T',
        y='value:Q',
        tooltip=['date:T', 'value:Q', 'shift_score:Q']
    )
    
    return (line + spikes + shifts).interactive()

def detect_spikes(series, z_score_threshold=3, window_size=7):
    """Detect spikes using Z-score method with tunable parameters"""
    rolling_mean = series.rolling(window=window_size, center=True).mean()
    rolling_std = series.rolling(window=window_size, center=True).std()
    z_scores = np.abs((series - rolling_mean) / rolling_std)
    
    spikes = z_scores > z_score_threshold
    spike_dates = series.index[spikes]
    spike_values = series[spikes]
    
    return pd.DataFrame({
        'date': spike_dates,
        'value': spike_values,
        'z_score': z_scores[spikes]
    }).sort_values('z_score', ascending=False)

def detect_level_shifts(series, shift_threshold=2, window_size=14):
    """Detect level shifts using rolling window comparison with tunable parameters"""
    before = series.rolling(window=window_size).mean().shift(1)
    after = series.rolling(window=window_size, center=True).mean()
    
    diff = np.abs(after - before)
    rolling_std = series.rolling(window=window_size*2).std()
    shift_scores = diff / rolling_std
    
    shifts = shift_scores > shift_threshold
    shift_dates = series.index[shifts]
    shift_values = series[shifts]
    
    return pd.DataFrame({
        'date': shift_dates,
        'value': shift_values,
        'shift_score': shift_scores[shifts]
    }).sort_values('shift_score', ascending=False)

def analyze_anomalies(data, date_column, metric_column, baseline_column=None, 
                     spike_params=None, shift_params=None):
    """Analyze anomalies with configurable parameters"""
    data = data.set_index(date_column)
    
    if baseline_column is not None:
        relative_metric = data[metric_column] / data[baseline_column]
        metrics_to_analyze = {
            f"{metric_column}": data[metric_column],
            f"{metric_column}_rel_to_{baseline_column}": relative_metric
        }
    else:
        metrics_to_analyze = {metric_column: data[metric_column]}
    
    results = {}
    for name, series in metrics_to_analyze.items():
        spikes = detect_spikes(series, **spike_params) if spike_params else detect_spikes(series)
        level_shifts = detect_level_shifts(series, **shift_params) if shift_params else detect_level_shifts(series)
        results[name] = {
            'spikes': spikes,
            'level_shifts': level_shifts
        }
    
    return results

# Streamlit App
def main():
    st.title("Time Series Anomaly Detection")
    st.write("Upload your CSV dataset and explore time series anomalies.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        data = pd.read_csv(uploaded_file)

        # Automatically identify categorical and numerical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = data.select_dtypes(include=['number']).columns.tolist()

        st.write("### Data Overview")
        st.write(data.head())
        st.write(f"Categorical Columns: {categorical_columns}")
        st.write(f"Numerical Columns: {numerical_columns}")

        # Sidebar Filters
        st.sidebar.write("### Data Filters")
        # Date Range Filter
        date_column = st.selectbox("Select Date Column", options=data.columns)
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data = data.dropna(subset=[date_column])
        date_min = data[date_column].min()
        date_max = data[date_column].max()
        date_range = st.sidebar.date_input("Select Date Range", [date_min, date_max], min_value=date_min, max_value=date_max)
        data = data[(data[date_column] >= pd.to_datetime(date_range[0])) & (data[date_column] <= pd.to_datetime(date_range[1]))]

        # Select the metric column and baseline column
        metric_column = st.selectbox("Select Metric Column for Analysis", options=numerical_columns)
        baseline_column = st.selectbox("Select Baseline Column (e.g., Visits for Relative Calculation)", options=[None] + numerical_columns)

        # Remove any rows with NaN in the metric or baseline column
        if baseline_column is not None:
            data = data.dropna(subset=[metric_column, baseline_column])
        else:
            data = data.dropna(subset=[metric_column])

        # Calculate top-10 categorical values and group the rest as 'Other'
        for col in categorical_columns:
            if col != date_column and col != metric_column:  # Make sure we are not applying grouping to the date or metric columns
                top_10_values = data.groupby(col)[metric_column].sum().nlargest(10).index.tolist()
                data[col] = data[col].apply(lambda x: x if x in top_10_values else 'Other')

        # Categorical Filters
        for col in categorical_columns:
            if col != date_column:  # Skip the date column for categorical filters
                unique_values = data[col].unique().tolist()
                selected_values = st.sidebar.multiselect(f"Filter {col}", options=unique_values, default=unique_values)
                if len(selected_values) < len(unique_values):
                    st.sidebar.write(f"Selected {len(selected_values)} out of {len(unique_values)} values for {col}")
                data = data[data[col].isin(selected_values)]

        # For each of categorical columns except date, Plot the factual layered area chart split by given column's value; put each chart into separate streamlit tab
        non_date_categorical_columns = [col for col in categorical_columns if col != date_column]
        tabs = st.tabs(non_date_categorical_columns)
        for i, col in enumerate(non_date_categorical_columns):
            with tabs[i]:
                st.write(f"### {col} Split Area Chart")
                area_chart_data = data.groupby([date_column, col])[metric_column].sum().reset_index()
                layered_chart = alt.Chart(area_chart_data).mark_area().encode(
                    x=alt.X(f'{date_column}:T', title='Date'),
                    y=alt.Y(f'{metric_column}:Q', title=metric_column),
                    color=alt.Color(f'{col}:N', title=col)
                ).properties(
                    width=800,
                    height=400
                )
                st.altair_chart(layered_chart, use_container_width=True)

        # Calculate the relative metric if baseline is provided
        if baseline_column is not None:
            relative_metric_column = f"{metric_column}_rel_to_{baseline_column}"
            # Aggregate metric and baseline by date before performing relative calculation
            aggregated_data = data.groupby(date_column).agg({metric_column: 'sum', baseline_column: 'sum'}).reset_index()
            aggregated_data[relative_metric_column] = aggregated_data[metric_column] / aggregated_data[baseline_column]
            st.write("### Relative Metric Calculation")
            st.write(aggregated_data[[date_column, metric_column, baseline_column, relative_metric_column]])

            # Option to select the metric to forecast
            metric_to_forecast = st.selectbox("Select Metric to Forecast", options=[metric_column, relative_metric_column])
        else:
            # Aggregate the metric column by date
            aggregated_data = data.groupby(date_column).agg({metric_column: 'sum'}).reset_index()
            metric_to_forecast = metric_column

            # Add sensitivity threshold slider
        st.sidebar.write("### Anomaly Detection Settings")
        sensitivity = st.sidebar.slider("Anomaly Detection Sensitivity", 1.0, 5.0, 3.0, 0.1)

                # Anomaly detection parameters
        st.sidebar.write("### Anomaly Detection Parameters")
        with st.sidebar.expander("Spike Detection"):
            spike_z_threshold = st.slider("Z-score Threshold", 1.0, 5.0, 3.0, 0.1)
            spike_window = st.slider("Window Size (days)", 3, 21, 7)

        with st.sidebar.expander("Level Shift Detection"):
            shift_threshold = st.slider("Shift Threshold", 1.0, 5.0, 2.0, 0.1)
            shift_window = st.slider("Window Size (days)", 7, 30, 14)


        # Update anomaly detection call
        anomalies = analyze_anomalies(
            aggregated_data,
            date_column,
            metric_to_forecast,
            baseline_column if baseline_column is not None else None,
            spike_params={'z_score_threshold': spike_z_threshold, 'window_size': spike_window},
            shift_params={'shift_threshold': shift_threshold, 'window_size': shift_window}
        )

        # Display anomalies
        st.write("### Detected Anomalies")
        for metric_name, results in anomalies.items():
            st.write(f"#### {metric_name}")
            
            spike_tab, shift_tab = st.tabs(["Spikes", "Level Shifts"])
            
            with spike_tab:
                if not results['spikes'].empty:
                    st.write("Top spikes detected:")
                    st.dataframe(results['spikes'])
                else:
                    st.write("No significant spikes detected")
                    
            with shift_tab:
                if not results['level_shifts'].empty:
                    st.write("Top level shifts detected:")
                    st.dataframe(results['level_shifts'])
                else:
                    st.write("No significant level shifts detected")

                    # Visualize anomalies
            st.write(f"### {metric_name} Time Series with Anomalies")
            chart = create_anomaly_chart(
                aggregated_data,
                date_column,
                metric_name,
                results,
                f"{metric_name} Anomalies"
            )
            st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()