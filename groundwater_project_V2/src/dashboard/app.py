import os
import json
import threading
import subprocess
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import joblib
import tensorflow as tf
from flask import (
    Flask, render_template, request, jsonify,
    send_file, flash, redirect, url_for, session, Response
)
from scipy.interpolate import griddata
import folium
from folium.plugins import MarkerCluster, HeatMap
import shap
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask
app = Flask(__name__)
app.secret_key = 'your-secure-random-key-here-change-in-production'

# ========== PATHS ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SPATIAL = os.path.join(BASE_DIR, '../../data/spatial/groundwater_spatial.csv')
MODEL_DIR = os.path.join(BASE_DIR, '../../models')
ALERTS_LOG = os.path.join(BASE_DIR, '../../data/alerts.csv')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
THRESHOLDS_FILE = os.path.join(BASE_DIR, '../../data/thresholds.json')

# ========== LOAD DATA ==========
df = pd.read_csv(DATA_SPATIAL, parse_dates=['timestamp'])
df['block'] = df['block'].fillna('Unknown').astype(str)

# Ensure alerts log exists
if not os.path.exists(ALERTS_LOG):
    pd.DataFrame(columns=['timestamp', 'well_id', 'parameter', 'value', 'threshold', 'message']).to_csv(ALERTS_LOG, index=False)

# Load model metrics if available
model_metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'r') as f:
        model_metrics = json.load(f)

# Load thresholds from session or file
def load_thresholds():
    if os.path.exists(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE, 'r') as f:
            return json.load(f)
    else:
        return {
            'water_level_low': 12,
            'water_level_high': 28,
            'ph_low': 7.0,
            'ph_high': 8.5,
            'tds_high': 3000,
            'ec_high': 5000,
            'turbidity_high': 50
        }

def save_thresholds(thresholds):
    with open(THRESHOLDS_FILE, 'w') as f:
        json.dump(thresholds, f, indent=4)

# Load models (lazy loading to speed up startup)
def load_keras_model(model_name):
    path = os.path.join(MODEL_DIR, model_name)
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

# ========== FEATURE LISTS ==========
time_features = ['water_level', 'ph', 'ec', 'tds', 'turbidity', 'temperature']
spatial_features = [col for col in df.columns if col.startswith('block_')] + ['dist_to_tiruppur_km', 'lat', 'lon']
all_features = time_features + spatial_features

# ========== WELL METADATA ==========
wells = df[['well_id', 'block', 'village', 'lat', 'lon']].drop_duplicates().to_dict('records')
blocks = sorted(df['block'].unique())
wells_by_block = {block: df[df['block'] == block][['well_id', 'village']].drop_duplicates().to_dict('records') for block in blocks}

# Latest readings per well
latest = df.sort_values('timestamp').groupby('well_id').last().reset_index()

# ========== HELPER FUNCTIONS ==========
def log_alert(well_id, parameter, value, threshold, message):
    alerts_df = pd.read_csv(ALERTS_LOG)
    new_row = pd.DataFrame({
        'timestamp': [datetime.now()],
        'well_id': [well_id],
        'parameter': [parameter],
        'value': [value],
        'threshold': [threshold],
        'message': [message]
    })
    alerts_df = pd.concat([alerts_df, new_row], ignore_index=True)
    alerts_df.to_csv(ALERTS_LOG, index=False)

def check_alerts(row, thresholds):
    alerts = []
    if row['water_level'] < thresholds['water_level_low']:
        msg = f"Low water level: {row['water_level']:.2f} m"
        alerts.append(msg)
        log_alert(row['well_id'], 'water_level', row['water_level'], thresholds['water_level_low'], msg)
    if row['water_level'] > thresholds['water_level_high']:
        msg = f"High water level: {row['water_level']:.2f} m"
        alerts.append(msg)
        log_alert(row['well_id'], 'water_level', row['water_level'], thresholds['water_level_high'], msg)
    if row['ph'] < thresholds['ph_low'] or row['ph'] > thresholds['ph_high']:
        msg = f"pH out of range: {row['ph']:.2f}"
        alerts.append(msg)
        log_alert(row['well_id'], 'ph', row['ph'], f"{thresholds['ph_low']}-{thresholds['ph_high']}", msg)
    if row['tds'] > thresholds['tds_high']:
        msg = f"High TDS: {row['tds']:.0f} mg/L"
        alerts.append(msg)
        log_alert(row['well_id'], 'tds', row['tds'], thresholds['tds_high'], msg)
    return alerts

# ========== ROUTES ==========
@app.route('/')
def index():
    thresholds = load_thresholds()
    # Compute aggregated alerts
    alert_counts = {}
    for _, row in latest.iterrows():
        if check_alerts(row, thresholds):
            block = row['block']
            alert_counts[block] = alert_counts.get(block, 0) + 1
    # Build map
    m = folium.Map(location=[11.0, 77.5], zoom_start=9, tiles='OpenStreetMap')
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in latest.iterrows():
        level = row['water_level']
        if level < thresholds['water_level_low']:
            color = 'red'
        elif level > thresholds['water_level_high']:
            color = 'darkred'
        else:
            color = 'blue'
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"<b>Well {row['well_id']}</b><br>Village: {row['village']}<br>Water Level: {level:.2f} m",
            tooltip=row['well_id'],
            icon=folium.Icon(color=color)
        ).add_to(marker_cluster)
    map_html = m._repr_html_()
    # Recent alerts
    alerts_df = pd.read_csv(ALERTS_LOG)
    recent_alerts = alerts_df.sort_values('timestamp', ascending=False).head(10).to_dict('records')
    return render_template('index.html',
                           map_html=map_html,
                           total_wells=len(wells),
                           avg_water_level=latest['water_level'].mean(),
                           wells_with_alerts=len(alert_counts),
                           blocks=blocks,
                           wells_by_block=wells_by_block,
                           recent_alerts=recent_alerts,
                           alert_counts=alert_counts,
                           last_update=datetime.now().strftime('%Y-%m-%d %H:%M'))

@app.route('/well/<int:well_id>')
def well_detail(well_id):
    well_data = df[df['well_id'] == well_id].sort_values('timestamp')
    if well_data.empty:
        return "Well not found", 404
    latest_row = well_data.iloc[-1]
    thresholds = load_thresholds()
    alerts = check_alerts(latest_row, thresholds)

    # ---- Anomaly detection using autoencoder ----
    autoencoder = load_keras_model('autoencoder_anomaly.h5')
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    anomaly_threshold = np.load(os.path.join(MODEL_DIR, 'anomaly_threshold.npy')) if os.path.exists(os.path.join(MODEL_DIR, 'anomaly_threshold.npy')) else 0.5
    recent = well_data.iloc[-12:][all_features].values
    anomaly_detected = False
    shap_html = None
    if len(recent) == 12 and autoencoder is not None:
        recent_scaled = recent.copy()
        recent_scaled[:, :len(time_features)] = scaler.transform(recent[:, :len(time_features)])
        recent_scaled = recent_scaled.reshape(1, 12, len(all_features))
        reconstruction = autoencoder.predict(recent_scaled)
        mse = np.mean(np.square(recent_scaled - reconstruction))
        if mse > anomaly_threshold:
            alerts.append(f"🚨 Anomaly detected (reconstruction error: {mse:.4f})")
            anomaly_detected = True
            # SHAP explanation (simplified)
            # For demo, we'll just create a placeholder
            shap_html = "<p>SHAP explanation would go here (requires background data).</p>"

    # ---- Prediction for next month ----
    cnn_lstm = load_keras_model('cnn_lstm_waterlevel.h5')
    pred = None
    if len(recent) == 12 and cnn_lstm is not None:
        X_pred = recent_scaled
        pred_scaled = cnn_lstm.predict(X_pred)[0, 0]
        dummy = np.zeros((1, len(time_features)))
        dummy[0, time_features.index('water_level')] = pred_scaled
        pred = scaler.inverse_transform(dummy)[0, time_features.index('water_level')]
        # Uncertainty (simulated)
        pred_std = pred * 0.1  # 10% uncertainty

    # ---- Plots ----
    figs = {}
    for param in time_features:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=well_data['timestamp'], y=well_data[param],
                                  mode='lines', name=param,
                                  line=dict(color='#007bff', width=2)))
        if param == 'water_level' and pred:
            # Add prediction point with uncertainty
            next_date = well_data['timestamp'].iloc[-1] + timedelta(days=30)
            fig.add_trace(go.Scatter(x=[next_date], y=[pred],
                                      mode='markers', name='Prediction',
                                      marker=dict(color='red', size=10)))
            # Uncertainty band
            fig.add_trace(go.Scatter(
                x=[next_date, next_date],
                y=[pred - 2*pred_std, pred + 2*pred_std],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                name='Uncertainty (±2σ)'
            ))
        fig.update_layout(title=f'{param.replace("_"," ").title()} over time',
                          xaxis_title='Date', yaxis_title=param,
                          template='plotly_white', height=400)
        figs[param] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # ---- Correlation scatter matrix ----
    corr_fig = go.Figure(data=go.Splom(
        dimensions=[dict(label=f, values=well_data[f]) for f in time_features],
        diagonal_visible=False
    ))
    corr_fig.update_layout(title='Parameter Correlations', height=600)

    # ---- Trend analysis (linear) ----
    from sklearn.linear_model import LinearRegression
    trend_figs = {}
    for param in time_features:
        y = well_data[param].dropna().values
        x = np.arange(len(y)).reshape(-1, 1)
        model_lr = LinearRegression().fit(x, y)
        trend = model_lr.predict(x)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=well_data['timestamp'][:len(y)], y=y, mode='lines', name='Original'))
        fig.add_trace(go.Scatter(x=well_data['timestamp'][:len(y)], y=trend, mode='lines', name=f'Trend (slope={model_lr.coef_[0]:.3f}/month)'))
        fig.update_layout(title=f'{param} Trend', template='plotly_white')
        trend_figs[param] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # ---- Seasonal decomposition (using statsmodels) ----
    from statsmodels.tsa.seasonal import seasonal_decompose
    seasonal_figs = {}
    for param in time_features:
        try:
            series = well_data[param].dropna()
            if len(series) >= 24:
                result = seasonal_decompose(series, model='additive', period=12)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series.index, y=result.trend, name='Trend'))
                fig.add_trace(go.Scatter(x=series.index, y=result.seasonal, name='Seasonal'))
                fig.add_trace(go.Scatter(x=series.index, y=result.resid, name='Residual'))
                fig.update_layout(title=f'{param} Decomposition', template='plotly_white')
                seasonal_figs[param] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except:
            pass

    return render_template('well_detail.html',
                           well=latest_row.to_dict(),
                           alerts=alerts,
                           pred=pred,
                           pred_std=pred_std if pred else None,
                           figs=figs,
                           corr_fig=json.dumps(corr_fig, cls=plotly.utils.PlotlyJSONEncoder),
                           trend_figs=trend_figs,
                           seasonal_figs=seasonal_figs,
                           anomaly_detected=anomaly_detected,
                           shap_html=shap_html,
                           well_id=well_id,
                           blocks=blocks,
                           wells_by_block=wells_by_block)

@app.route('/maps')
def maps_page():
    months = sorted(df['timestamp'].dt.to_period('M').unique())
    month_strings = [str(m) for m in months]
    return render_template('maps.html',
                           months=month_strings,
                           blocks=blocks,
                           wells_by_block=wells_by_block,
                           wells=wells)   # <-- add this line

@app.route('/api/interpolated/<month>')
def api_interpolated(month):
    try:
        target = pd.to_datetime(month)
    except:
        return jsonify({'error': 'Invalid month format'}), 400

    window = timedelta(days=15)
    mask = (df['timestamp'] >= target - window) & (df['timestamp'] <= target + window)
    month_data = df[mask].groupby('well_id').mean().reset_index()

    if month_data.empty:
        print(f"No data for {month}")  # Debug log
        return jsonify([])  # Return empty list

    points = month_data[['lon', 'lat']].values
    values = month_data['water_level'].values

    if len(points) < 4:
        print(f"Not enough points for interpolation in {month}: {len(points)}")
        return jsonify([])

    # Create grid
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(df['lon'].min() - 0.1, df['lon'].max() + 0.1, 50),
        np.linspace(df['lat'].min() - 0.1, df['lat'].max() + 0.1, 50)
    )
    grid = griddata(points, values, (lon_grid, lat_grid), method='cubic')

    heat_data = []
    for i in range(lat_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            if not np.isnan(grid[i, j]):
                # Leaflet heat expects [lat, lon, intensity]
                heat_data.append([lat_grid[i, j], lon_grid[i, j], grid[i, j]])

    print(f"Returning {len(heat_data)} heat points for {month}")  # Debug
    return jsonify(heat_data)

@app.route('/compare')
def compare_page():
    well_ids = request.args.getlist('wells')
    well_ids = [int(i) for i in well_ids if i.isdigit()]
    graphs = {}
    if well_ids:
        selected = df[df['well_id'].isin(well_ids)]
        for param in time_features:
            fig = go.Figure()
            for wid in well_ids:
                wdata = selected[selected['well_id'] == wid]
                fig.add_trace(go.Scatter(x=wdata['timestamp'], y=wdata[param],
                                          mode='lines', name=f"Well {wid}"))
            fig.update_layout(title=param, template='plotly_white')
            graphs[param] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('compare.html',
                           graphs=graphs,
                           all_wells=wells,
                           blocks=blocks,
                           wells_by_block=wells_by_block)

@app.route('/models')
def models_page():
    global model_metrics
    # Reload metrics from file if it exists
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            model_metrics = json.load(f)
    else:
        model_metrics = {}
    return render_template('models.html',
                           metrics=model_metrics,
                           blocks=blocks,
                           wells_by_block=wells_by_block,
                           wells=wells)   # Pass wells for dropdowns

@app.route('/retrain', methods=['POST'])
def retrain():
    def train_task():
        import sys
        import subprocess
        # Path to model_building.py
        script_path = os.path.join(BASE_DIR, '../model_building.py')
        log_path = os.path.join(BASE_DIR, '../../data/retrain_log.txt')
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as log:
            result = subprocess.run(
                [sys.executable, script_path],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(script_path)  # Set working directory to src/
            )
        # After training, reload metrics
        global model_metrics
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                model_metrics = json.load(f)
    threading.Thread(target=train_task).start()
    flash('Retraining started in background. Check data/retrain_log.txt for progress.', 'success')
    return redirect(url_for('models_page'))

@app.route('/alerts')
def alerts_page():
    alerts_df = pd.read_csv(ALERTS_LOG)
    if not alerts_df.empty:
        # Ensure well_id is integer
        alerts_df['well_id'] = alerts_df['well_id'].astype(int)
        # Merge with well info to get block
        wells_info = df[['well_id', 'block']].drop_duplicates()
        alerts_df = alerts_df.merge(wells_info, on='well_id', how='left')
        # Aggregated alerts by block and parameter
        agg = alerts_df.groupby(['block', 'parameter']).size().reset_index(name='count')
        agg = agg.to_dict('records')
    else:
        agg = []
    alerts = alerts_df.sort_values('timestamp', ascending=False).to_dict('records')
    thresholds = load_thresholds()
    return render_template('alerts.html',
                           alerts=alerts,
                           thresholds=thresholds,
                           agg=agg,
                           blocks=blocks,
                           wells_by_block=wells_by_block)

@app.route('/alerts/settings', methods=['POST'])
def alerts_settings():
    thresholds = {
        'water_level_low': float(request.form.get('water_level_low', 12)),
        'water_level_high': float(request.form.get('water_level_high', 28)),
        'ph_low': float(request.form.get('ph_low', 7.0)),
        'ph_high': float(request.form.get('ph_high', 8.5)),
        'tds_high': float(request.form.get('tds_high', 3000)),
        'ec_high': float(request.form.get('ec_high', 5000)),
        'turbidity_high': float(request.form.get('turbidity_high', 50))
    }
    save_thresholds(thresholds)
    flash('Alert thresholds updated.', 'success')
    return redirect(url_for('alerts_page'))

@app.route('/data')
def data_page():
    return render_template('data.html',
                           wells=wells,
                           blocks=blocks,
                           wells_by_block=wells_by_block)

@app.route('/data/export', methods=['POST'])
def export_data():
    well_ids = request.form.getlist('wells')
    well_ids = [int(i) for i in well_ids if i.isdigit()]
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    filtered = df.copy()
    if well_ids:
        filtered = filtered[filtered['well_id'].isin(well_ids)]
    if start_date:
        filtered = filtered[filtered['timestamp'] >= start_date]
    if end_date:
        filtered = filtered[filtered['timestamp'] <= end_date]
    csv = filtered.to_csv(index=False)
    return Response(csv, mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment;filename=groundwater_data.csv'})

@app.route('/data/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('data_page'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('data_page'))
    if file and file.filename.endswith('.csv'):
        try:
            new_df = pd.read_csv(file, parse_dates=['timestamp'])
            # Basic validation: required columns
            required = ['well_id', 'timestamp', 'water_level', 'ph', 'ec', 'tds', 'turbidity', 'temperature']
            if all(col in new_df.columns for col in required):
                # Append to existing spatial data (you may want to preprocess first)
                new_df.to_csv(DATA_SPATIAL, mode='a', header=False, index=False)
                flash(f'Uploaded {len(new_df)} rows successfully.', 'success')
            else:
                flash('CSV missing required columns', 'danger')
        except Exception as e:
            flash(f'Error processing file: {e}', 'danger')
    else:
        flash('Only CSV files allowed', 'danger')
    return redirect(url_for('data_page'))

@app.route('/scenarios')
def scenarios_page():
    return render_template('scenarios.html',
                           wells=wells,
                           blocks=blocks,
                           wells_by_block=wells_by_block)

@app.route('/api/scenario', methods=['POST'])
def scenario_api():
    data = request.get_json()
    well_id = data.get('well_id')
    pumping = data.get('pumping', 1.0)
    rainfall = data.get('rainfall', 1.0)
    # Simple linear model: water_level = base + a*pumping + b*rainfall
    # For demo, we use precomputed coefficients from a quick regression
    # In reality, you'd have a trained model per well.
    well_data = df[df['well_id'] == well_id].sort_values('timestamp')
    if well_data.empty:
        return jsonify({'error': 'Well not found'}), 404
    base = well_data['water_level'].iloc[-1]
    # Dummy coefficients
    effect = -2.5 * pumping + 1.2 * rainfall
    predicted = base + effect
    return jsonify({'predicted': predicted})

@app.route('/report/<int:well_id>')
def generate_report(well_id):
    from weasyprint import HTML
    well_data = df[df['well_id'] == well_id].sort_values('timestamp')
    if well_data.empty:
        return "Well not found", 404
    latest_row = well_data.iloc[-1]
    # Create a few graphs as base64 images (simplified: use plotly to_html and embed)
    # For brevity, we'll just render a simple HTML template.
    html_string = render_template('report_template.html',
                                  well=latest_row.to_dict(),
                                  data=well_data.head(100))  # last 100 rows
    pdf = HTML(string=html_string).write_pdf()
    return Response(pdf, mimetype='application/pdf',
                    headers={'Content-Disposition': f'attachment;filename=well_{well_id}_report.pdf'})

if __name__ == '__main__':
    app.run(debug=True)