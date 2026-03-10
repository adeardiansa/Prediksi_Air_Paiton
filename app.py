import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
import json
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Riwayat dan Prediksi ARIMA", layout="wide")
st.title("Tampilan Data Riwayat Excel dengan Prediksi ARIMA")

# Sidebar
# Folder tempat file excel berada
data_folder = 'data'  # pastikan folder ini ada di direktori project

# Ambil semua file .xlsx dalam folder
excel_files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]

selected_file = st.sidebar.selectbox("Pilih file Excel dari direktori:", excel_files)

# Fungsi membaca dan membersihkan Excel
def read_clean_excel(xls, sheet_name):
    raw_df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    for idx, row in raw_df.iterrows():
        if row.notna().sum() > 2 and 'Tanggal' in row.values:
            header_row = idx
            break
    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)
    
    if 'Tanggal' in df.columns:
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce').dt.date

    for col in df.select_dtypes(include='number').columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

    return df

# Fungsi untuk menentukan parameter ARIMA otomatis
def find_best_arima_params(ts, max_p=3, max_d=2, max_q=3):
    best_aic = float('inf')
    best_params = None
    
    adf_result = adfuller(ts.dropna())
    d = 0 if adf_result[1] <= 0.05 else 1
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(ts, order=(p, d, q))
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_params = (p, d, q)
            except:
                continue
    
    return best_params if best_params else (1, 1, 1)

# Fungsi prediksi dengan ARIMA + titik simulasi + MAE/MSE
def arima_forecast_with_simulation(df, target_col, future_dates, num_points=20, test_size=0.2):
    df_clean = df[['Tanggal', target_col]].dropna()
    df_clean = df_clean[df_clean['Tanggal'].notna()]
    df_clean = df_clean.sort_values(by='Tanggal')

    if len(df_clean) < 5:
        return [None] * len(future_dates), [], None, None

    ts = pd.Series(df_clean[target_col].values, index=pd.to_datetime(df_clean['Tanggal']))

    try:
        # Cari parameter ARIMA terbaik
        best_params = find_best_arima_params(ts)
        
        # Pisahkan data menjadi pelatihan dan pengujian
        train_size = int(len(ts) * (1 - test_size))
        train_ts = ts[:train_size]
        test_ts = ts[train_size:]
        
        # Fit model ARIMA pada data pelatihan
        model = ARIMA(train_ts, order=best_params)
        fitted_model = model.fit()
        
        # Prediksi pada data pengujian
        test_forecast = fitted_model.forecast(steps=len(test_ts))
        
        # Hitung MAE dan MSE
        mae = mean_absolute_error(test_ts, test_forecast)
        mse = mean_squared_error(test_ts, test_forecast)
        
        # Fit model ARIMA pada seluruh data untuk prediksi masa depan
        model = ARIMA(ts, order=best_params)
        fitted_model = model.fit()
        
        # Prediksi untuk tanggal masa depan
        forecast_result = fitted_model.forecast(steps=len(future_dates), alpha=0.05)
        forecast_values = forecast_result
        
        # Hitung residuals untuk simulasi
        residuals = fitted_model.resid
        std_dev = residuals.std()

        final_preds = []
        scatter_points = []

        for idx, val in enumerate(forecast_values):
            date_point = future_dates[idx]
            
            # Simulasi Monte Carlo
            simulated_points = np.random.normal(loc=val, scale=std_dev, size=num_points)
            
            for sim in simulated_points:
                scatter_points.append({"Tanggal": date_point, "Nilai": round(sim, 2)})

            # Gunakan median dari simulasi sebagai prediksi final
            mean_simulated = np.median(simulated_points)
            final_preds.append(round(mean_simulated, 2))

        return final_preds, scatter_points, round(mae, 2), round(mse, 2)
    except Exception as e:
        st.warning(f"Error dalam fitting ARIMA untuk {target_col}: {str(e)}")
        return [None] * len(future_dates), [], None, None

# Fungsi untuk membuat JSON hasil prediksi
def create_prediction_json(pred_df, mae_values, mse_values, selected_file, selected_sheet, start_date, end_date):
    json_data = {
        "metadata": {
            "file_name": selected_file,
            "sheet_name": selected_sheet,
            "prediction_start_date": str(start_date),
            "prediction_end_date": str(end_date),
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "ARIMA"
        },
        "evaluation_metrics": {},
        "predictions": []
    }
    
    # Tambahkan metrik evaluasi
    for col, mae in mae_values.items():
        json_data["evaluation_metrics"][col] = {
            "MAE": mae if mae is not None else "N/A",
            "MSE": mse_values[col] if mse_values[col] is not None else "N/A"
        }
    
    # Konversi prediksi ke format JSON
    for _, row in pred_df.iterrows():
        prediction_row = {"Tanggal": str(row["Tanggal"])}
        for col in pred_df.columns:
            if col != "Tanggal":
                prediction_row[col] = row[col] if pd.notna(row[col]) else None
        json_data["predictions"].append(prediction_row)
    
    return json_data

# Main App
if selected_file:
    xls = pd.ExcelFile(os.path.join(data_folder,selected_file))
    sheet_names = xls.sheet_names
    selected_sheet = st.sidebar.selectbox("Pilih sheet:", sheet_names)

    try:
        df = read_clean_excel(xls, selected_sheet)
        st.subheader(f"Data Sheet: {selected_sheet}")
        st.dataframe(df, use_container_width=True)

        numeric_columns = [col for col in df.select_dtypes(include='number').columns]

        st.markdown("---")
        st.subheader("Prediksi Rentang Tanggal dengan Model ARIMA")

        date_range = st.date_input("Pilih rentang tanggal:", value=[date.today(), date.today() + timedelta(days=3)])

        if len(date_range) == 2:
            start_date, end_date = date_range
            if start_date > end_date:
                st.warning("Tanggal awal tidak boleh lebih besar dari tanggal akhir.")
            elif st.button("Prediksi Rentang Tanggal"):
                future_dates = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
                predictions = [{"Tanggal": d.date()} for d in future_dates]

                scatter_all = []
                mae_values = {}
                mse_values = {}

                for col in numeric_columns:
                    result = arima_forecast_with_simulation(
                        df, col, [d.date() for d in future_dates]
                    )
                    
                    if len(result) == 4:
                        preds, scatter_points, mae, mse = result
                        mae_values[col] = mae
                        mse_values[col] = mse
                    else:
                        preds, scatter_points = result
                        mae_values[col] = None
                        mse_values[col] = None
                    
                    for i in range(len(predictions)):
                        predictions[i][col] = preds[i]
                    
                    for pt in scatter_points:
                        pt['Aspek'] = col
                        scatter_all.append(pt)

                pred_df = pd.DataFrame(predictions)
                
                # Tampilkan metrik evaluasi
                st.subheader("Metrik Evaluasi")
                param_info = []
                for col in numeric_columns:
                    param_info.append({
                        "Kolom": col,
                        "MAE": mae_values[col] if mae_values[col] is not None else "N/A",
                        "MSE": mse_values[col] if mse_values[col] is not None else "N/A"
                    })
                
                if param_info:
                    st.dataframe(pd.DataFrame(param_info), use_container_width=True)
                
                st.subheader("Tabel Prediksi dengan Simulasi ARIMA")
                st.dataframe(pred_df, use_container_width=True)

                # Tombol Download JSON
                st.subheader("Download Hasil Prediksi")
                json_data = create_prediction_json(pred_df, mae_values, mse_values, selected_file, selected_sheet, start_date, end_date)
                json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_string,
                        file_name=f"prediksi_arima_{selected_sheet}_{start_date}_{end_date}.json",
                        mime="application/json",
                        help="Download hasil prediksi dalam format JSON"
                    )
                
                with col2:
                    st.info("File JSON berisi metadata, metrik evaluasi, dan hasil prediksi lengkap")

                st.subheader("Grafik Gabungan: Data Historis + Prediksi ARIMA")
                scatter_df = pd.DataFrame(scatter_all)

                for col in numeric_columns:
                    if col not in pred_df.columns:
                        continue
                    
                    fig = go.Figure()
                    
                    # Ambil data historis untuk kolom ini
                    hist_data = df[['Tanggal', col]].dropna()
                    hist_data = hist_data[hist_data['Tanggal'].notna()]
                    hist_data = hist_data.sort_values(by='Tanggal')
                    
                    # Data historis
                    if not hist_data.empty:
                        fig.add_trace(go.Scatter(
                            x=hist_data['Tanggal'],
                            y=hist_data[col],
                            mode="lines+markers",
                            name=f"Data Historis {col}",
                            line=dict(color="green", width=2),
                            marker=dict(size=6, color="green")
                        ))
                    
                    # Titik simulasi prediksi
                    scatter_filtered = scatter_df[scatter_df["Aspek"] == col]
                    if not scatter_filtered.empty:
                        fig.add_trace(go.Scatter(
                            x=scatter_filtered["Tanggal"],
                            y=scatter_filtered["Nilai"],
                            mode="markers",
                            marker=dict(size=4, color='lightcoral', opacity=0.6),
                            name="Simulasi"
                        ))

                    # Prediksi utama
                    fig.add_trace(go.Scatter(
                        x=pred_df["Tanggal"],
                        y=pred_df[col],
                        mode="lines+markers",
                        name=f"Prediksi ARIMA {col}",
                        line=dict(color="red", width=3, dash="dash"),
                        marker=dict(size=8, color="red")
                    ))

                    # Garis pemisah antara historis dan prediksi
                    if not hist_data.empty:
                        last_historical_date = hist_data['Tanggal'].iloc[-1]
                        last_historical_value = hist_data[col].iloc[-1]
                        first_prediction_date = pred_df["Tanggal"].iloc[0]
                        first_prediction_value = pred_df[col].iloc[0]
                        
                        # Garis penghubung
                        # fig.add_trace(go.Scatter(
                        #     x=[last_historical_date, first_prediction_date],
                        #     y=[last_historical_value, first_prediction_value],
                        #     mode="lines",
                        #     line=dict(color="orange", width=2, dash="dot"),
                        #     name="Transisi",
                        #     showlegend=True
                        # ))
                        
                        # Garis vertikal pemisah
                        y_range = [
                            min(hist_data[col].min(), pred_df[col].min()) - 5,
                            max(hist_data[col].max(), pred_df[col].max()) + 5
                        ]
                        fig.add_shape(
                            type="line",
                            x0=last_historical_date, y0=y_range[0],
                            x1=last_historical_date, y1=y_range[1],
                            line=dict(color="gray", width=2, dash="dashdot")
                        )
                        
                        # Annotasi pemisah
                        fig.add_annotation(
                            x=last_historical_date,
                            y=y_range[1],
                            text="Batas Historis",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="gray",
                            bgcolor="white",
                            bordercolor="gray"
                        )

                    # Tambahkan informasi statistik, MAE, dan MSE
                    mae_mse_str = f"MAE={mae_values[col]:.2f}, MSE={mse_values[col]:.2f}" if mae_values[col] is not None else "MAE=N/A, MSE=N/A"
                    
                    if not hist_data.empty:
                        hist_mean = hist_data[col].mean()
                        hist_std = hist_data[col].std()
                        pred_mean = pred_df[col].mean()
                        stats_text = f"Historis: μ={hist_mean:.2f}, σ={hist_std:.2f} | Prediksi: μ={pred_mean:.2f} | {mae_mse_str}"
                    else:
                        stats_text = mae_mse_str
                    
                    fig.update_layout(
                        title=f"Perbandingan Data : {col}",
                        xaxis_title="Tanggal",
                        yaxis_title="Nilai",
                        hovermode="x unified",
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal membaca sheet: {e}")