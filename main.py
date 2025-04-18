import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import random
from io import BytesIO
import base64
import folium
from streamlit_folium import folium_static
from streamlit.components.v1 import html

# Set page config
st.set_page_config(
    page_title="OASYS Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
/* -- WARNA & TEMA UMUM -- */
body, .main, .block-container {
    background-color: #FFFDF5;
    font-family: 'Poppins', sans-serif;
    color: #2C3E50;
}

/* -- HEADER UTAMA -- */
h1, h2, h3, h4 {
    color: #FF8C00;
    font-weight: 700;
}

/* -- SIDEBAR -- */
[data-testid="stSidebar"] {
    background-color: #FFF3CD;
    color: #D35400;
}

[data-testid="stSidebar"] .sidebar-content {
    padding: 2rem 1rem;
}

.sidebar-brand {
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    padding: 10px;
    background-color: #FFEB99;
    border-radius: 8px;
    color: #FF8C00;
    margin-bottom: 5px;
}

.sidebar-brand-subtitle {
    font-size: 16px;
    text-align: center;
    font-style: italic;
    color: #FF8C00;
    # color: #e67e22;
    margin-bottom: 20px;
}

/* -- METRICS / ANGKA BESAR -- */
.stMetric {
    background-color: #FFF8E1;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(255, 140, 0, 0.1);
    color: #FF8C00;
    text-align: left;
}

/* -- KONTEN UTAMA -- */
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    border-left: 6px solid #FFA500;
    box-shadow: 0 4px 8px rgba(0,0,0,0.07);
    margin-bottom: 20px;
    font-size: 16px;
    line-height: 1.6;
}
            
 .success-story {
    padding: 15px;
    border-radius: 12px;
    background-color: #f0f9ff;
    margin-bottom: 15px;
 }
            
 .testimonial {
    font-style: italic;
    border-left: 4px solid #3498db;
    padding-left: 12px;
 }
            
 .tips-card {
    background-color: #e8f4ea;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
 }

ul li::marker {
    color: #FF8C00;
}

/* -- MAP & KOMPONEN FULL WIDTH -- */
.full-width-map {
    width: 100% !important;
    height: 500px !important;
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid #FFA500;
}

/* -- TOOLTIP/INPUT SELECT -- */
.css-1d391kg {
    color: #2C3E50 !important;
    font-size: 14px !important;

.cc {
    background-color: #e8f4ea;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)



# ----- FUNCTIONS FOR DATA GENERATION -----

def generate_demo_data():
    """Generate dummy data for the dashboard"""
    
    # List of kecamatan in Jakarta
    kecamatan_list = [
        "Cakung", "Cilincing", "Kelapa Gading", "Koja", "Pademangan", "Tanjung Priok",  # Jakarta Utara
        "Cempaka Putih", "Gambir", "Johar Baru", "Kemayoran", "Menteng", "Sawah Besar", "Senen", "Tanah Abang",  # Jakarta Pusat
        "Cilandak", "Jagakarsa", "Kebayoran Baru", "Kebayoran Lama", "Mampang Prapatan", "Pancoran", "Pasar Minggu", "Pesanggrahan", "Setia Budi", "Tebet",  # Jakarta Selatan
        "Cengkareng", "Grogol Petamburan", "Kalideres", "Kebon Jeruk", "Kembangan", "Palmerah", "Taman Sari", "Tambora",  # Jakarta Barat
        "Cipayung", "Ciracas", "Duren Sawit", "Jatinegara", "Kramat Jati", "Makasar", "Matraman", "Pasar Rebo"  # Jakarta Timur
    ]
    
    # Generate waste collection data
    today = datetime(2025, 4, 18)  # Set to match the current date
    dates = [today - timedelta(days=x) for x in range(90)]
    
    # Simplified waste types as requested
    waste_types = ["Organik", "Recycle-able"]

    waste_collection_data = []
    
    for date in dates:
        for kecamatan in kecamatan_list:
            for waste_type in waste_types:
                # Random factors to create variation in data
                population_factor = random.uniform(0.8, 1.2)
                seasonal_factor = 1 + 0.2 * np.sin(date.month / 12 * 2 * np.pi)
                weekend_factor = 1.3 if date.weekday() >= 5 else 1.0
                
                # Base weights for different waste types
                base_weights = {
                    "Organik": 60,
                    "Recycle-able": 40
                }
                
                # Calculate weight
                weight = (
                    base_weights[waste_type] * 
                    population_factor * 
                    seasonal_factor * 
                    weekend_factor * 
                    random.uniform(0.7, 1.3)
                )
                
                # Status (80% already distributed, 20% still in TPS)
                status = "Disalurkan" if random.random() < 0.8 else "Di TPS"
                
                waste_collection_data.append({
                    "date": date.date(),
                    "time": f"{random.randint(7, 18)}:{random.choice(['00', '15', '30', '45'])}",
                    "kecamatan": kecamatan,
                    "waste_type": waste_type,
                    "weight_kg": round(weight, 2),
                    "status": status
                })
    
    # Create DataFrame
    df = pd.DataFrame(waste_collection_data)
    
    # Add bins data
    bins_data = []
    total_bins = 1200
    
    for kecamatan in kecamatan_list:
        # Allocate bins based on a random distribution
        kecamatan_bins = int(total_bins * random.uniform(0.01, 0.05))
        full_bins = int(kecamatan_bins * random.uniform(0.1, 0.4))
        
        bins_data.append({
            "kecamatan": kecamatan,
            "total_bins": kecamatan_bins,
            "full_bins": full_bins,
            "active_bins": kecamatan_bins - int(kecamatan_bins * random.uniform(0, 0.1))  # Some bins might be inactive
        })
    
    bins_df = pd.DataFrame(bins_data)
    
    # Add user data
    users_data = []
    total_users = 9500
    
    # Monthly growth over the past 6 months
    months = [today - timedelta(days=30*x) for x in range(6)]
    months.reverse()  # Start from the oldest
    
    cumulative_users = int(total_users * 0.4)  # Start with 40% of current users
    
    for i, month in enumerate(months):
        growth_rate = random.uniform(0.1, 0.25)
        new_users = int(cumulative_users * growth_rate)
        cumulative_users += new_users
        
        if i == len(months) - 1:  # Make sure we end up with the total
            cumulative_users = total_users
            
        users_data.append({
            "month": month.strftime("%b %Y"),
            "users": cumulative_users
        })
    
    users_df = pd.DataFrame(users_data)
    
    # Kecamatan geospatial coordinates (approximated)
    # This is a simplified version - real implementation would use actual geo coordinates
    geo_data = {}
    
    latitude_base = -6.2
    longitude_base = 106.8
    
    for i, kecamatan in enumerate(kecamatan_list):
        geo_data[kecamatan] = {
            "lat": latitude_base + random.uniform(-0.1, 0.1),
            "lon": longitude_base + random.uniform(-0.1, 0.1)
        }
    
    return df, bins_df, users_df, geo_data, kecamatan_list

# Load or generate data
df, bins_df, users_df, geo_data, kecamatan_list = generate_demo_data()

# ----- SIDEBAR -----
st.sidebar.markdown("""
<div class="sidebar-brand">
    OASYS
</div>
<div class="sidebar-brand-subtitle">
    Optimized Automated System for Sustainable Waste
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Filter Data")

# Date filter
date_range = st.sidebar.date_input(
    "Pilih Rentang Waktu",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Kecamatan filter
selected_kecamatan = st.sidebar.multiselect(
    "Pilih Kecamatan",
    options=kecamatan_list,
    default=[]
)

# Waste type filter
waste_types = df['waste_type'].unique().tolist()
selected_waste_types = st.sidebar.multiselect(
    "Pilih Jenis Sampah",
    options=waste_types,
    default=[]
)

# Apply filters
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
else:
    df_filtered = df

if selected_kecamatan:
    df_filtered = df_filtered[df_filtered['kecamatan'].isin(selected_kecamatan)]

if selected_waste_types:
    df_filtered = df_filtered[df_filtered['waste_type'].isin(selected_waste_types)]

# If no filters are selected, use all data
if not selected_kecamatan and not selected_waste_types:
    df_filtered = df_filtered

# ----- MAIN DASHBOARD -----

# Header
st.title("♻️ OASYS - Dashboard Pengelolaan Sampah Jakarta")
st.markdown('<p style="font-size:24px; color:#FF8C00; font-style:italic;">Optimized Automated System for Sustainable Waste</p>', unsafe_allow_html=True)

#  background-color: #ffffff;
#     
#     
#     
#     
#     margin-bottom: 20px;
#     font-size: 16px;
#     line-height: 1.6;
# Explainer text
st.markdown("""
    <style>
    /* Style khusus untuk semua expander judul */
    .streamlit-expanderHeader {
        font-family: 'Segoe UI';
        font-size: 20px;
        font-weight: 700;
        color: #2C3E50;
    }
    </style>
""", unsafe_allow_html=True)

with st.expander("Mengapa Pengelolaan Sampah Penting?"):
    st.markdown("""
    <div class="card">
        <h3>Dampak Sampah di Jakarta</h3>
        <p>Jakarta menghasilkan lebih dari <b>7.500 ton sampah setiap hari</b>, dengan sebagian besar berakhir di TPA Bantargebang yang hampir mencapai kapasitas maksimal. Pengelolaan sampah yang tepat dapat:</p>
        <ul>
            <li>Mengurangi emisi gas rumah kaca</li>
            <li>Menciptakan lingkungan yang lebih bersih dan sehat</li>
            <li>Menghasilkan nilai ekonomi dari sampah</li>
            <li>Memperpanjang umur TPA yang ada</li>
        </ul>
        <p>Melalui sistem kami, kami berupaya menciptakan ekosistem pengelolaan sampah yang terpadu dan berkelanjutan di Jakarta.</p>
    </div>
    """, unsafe_allow_html=True)

# Key metrics
col1, col2, col3 = st.columns(3)

# Calculate metrics
total_waste = df_filtered['weight_kg'].sum()
distributed_waste = df_filtered[df_filtered['status'] == 'Disalurkan']['weight_kg'].sum()
total_active_bins = bins_df['active_bins'].sum()

col1.metric("Total Sampah Diterima", f"{total_waste:,.2f} kg")
col2.metric("Total Sampah Disalurkan", f"{distributed_waste:,.2f} kg", f"{(distributed_waste/total_waste*100):.1f}%")
col3.metric("Tempat Sampah Aktif", f"{total_active_bins:,}")

# Create two rows for main content
row1_col1, row1_col2 = st.columns([2, 1])

# ROW 1: Map and Impact tracker
with row1_col1:
    st.markdown("<div class='card'><h3>Peta Sebaran Sampah per Kecamatan</h3>", unsafe_allow_html=True)
    
    # Create a base map centered on Jakarta
    m = folium.Map(location=[-6.2, 106.8], zoom_start=11)
    
    # Calculate waste per kecamatan
    waste_per_kecamatan = df_filtered.groupby('kecamatan')['weight_kg'].sum().reset_index()
    
    # Add circles for each kecamatan
    for _, row in waste_per_kecamatan.iterrows():
        kecamatan = row['kecamatan']
        weight = row['weight_kg']
        
        # Skip if we don't have geo data for this kecamatan
        if kecamatan not in geo_data:
            continue
            
        # Get coordinates
        lat = geo_data[kecamatan]['lat']
        lon = geo_data[kecamatan]['lon']
        
        # Scale the radius based on weight
        radius = np.sqrt(weight) * 20
        
        # Scale color based on weight - REVERSED as requested (large=green, small=red)
        # Create bins for color mapping
        max_weight = waste_per_kecamatan['weight_kg'].max()
        normalized_weight = weight / max_weight
        
        # Color ranges from red (low) to green (high) - REVERSED from original
        r = int(255 * (1 - normalized_weight))
        g = int(255 * normalized_weight)
        b = 0
        
        color = f'#{r:02x}{g:02x}{b:02x}'
        
        folium.Circle(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            tooltip=f"{kecamatan}: {weight:.2f} kg"
        ).add_to(m)
        
        # Add kecamatan name
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                icon_size=(150, 36),
                icon_anchor=(75, 18),
                html=f'<div style="font-size: 10pt; color: black; text-align: center;"><b>{kecamatan}</b></div>'
            )
        ).add_to(m)
    
    # Display the map
    # folium_static(m)
    # st.markdown('<div class="full-width-map">', unsafe_allow_html=True)
    # folium_static(m)
    html(m.get_root().render(), height=500, width=0)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with row1_col2:
    st.markdown("<div class='card'><h3>Impact Tracker</h3>", unsafe_allow_html=True)
    
    # Calculate impact metrics
    organik_weight = df_filtered[df_filtered['waste_type'] == 'Organik']['weight_kg'].sum()
    recyclable_weight = df_filtered[df_filtered['waste_type'] == 'Recycle-able']['weight_kg'].sum()
    
    # Assuming recyclable is a mix of plastic (50%), paper (30%), and metal (20%)
    plastik_weight = recyclable_weight * 0.5
    kertas_weight = recyclable_weight * 0.3
    logam_weight = recyclable_weight * 0.2
    
    # CO2 saved calculations
    co2_organik = organik_weight * 0.5  # 0.5 kg CO₂ per kg organic waste
    co2_plastik = plastik_weight * 2.5  # 2.5 kg CO₂ per kg plastic
    co2_kertas = kertas_weight * 0.8    # 0.8 kg CO₂ per kg paper
    co2_logam = logam_weight * 9.0      # 9 kg CO₂ per kg aluminum
    
    total_co2_saved = co2_organik + co2_plastik + co2_kertas + co2_logam
    
    # Trees saved
    trees_saved = kertas_weight / 1000 * 17  # 17 trees per ton of paper
    
    # Water saved
    water_saved_paper = kertas_weight / 1000 * 7000  # 7000 liters per ton of paper
    water_saved_plastic = plastik_weight / 1000 * 16000  # 16000 liters per ton of plastic
    total_water_saved = water_saved_paper + water_saved_plastic
    
    # Display metrics
    st.info(f"Total Karbon Terhindar: **{total_co2_saved:,.2f} kg CO₂**")
    
    st.markdown("""
    <ul style="font-size:0.9em; margin-top:-10px;">
        <li>Sampah organik: 0.5 kg CO₂ per kg sampah</li>
        <li>Plastik: 2.5 kg CO₂ per kg plastik</li>
        <li>Kertas: 0.8 kg CO₂ per kg kertas</li>
        <li>Logam: 9 kg CO₂ per kg aluminium</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.info(f"Pohon Terselamatkan: **{trees_saved:,.1f} pohon**")
    st.markdown("""
    <p style="font-size:0.9em; margin-top:-10px;">
        Berdasarkan estimasi 17 pohon terselamatkan per ton kertas yang didaur ulang
    </p>
    """, unsafe_allow_html=True)
    
    st.info(f"Air Terselamatkan: **{total_water_saved:,.1f} liter**")
    st.markdown("""
    <ul style="font-size:0.9em; margin-top:-10px;">
        <li>7,000 liter air per ton kertas didaur ulang</li>
        <li>16,000 liter air per ton plastik didaur ulang</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ROW 2: Graphs and Community engagement
row2_col1, row2_col2 = st.columns([3, 2])

with row2_col1:
    st.markdown("<div class='card'><h3>Komposisi Sampah</h3>", unsafe_allow_html=True)
    
    # Prepare data
    waste_composition = df_filtered.groupby('waste_type')['weight_kg'].sum().reset_index()
    waste_composition = waste_composition.sort_values('weight_kg', ascending=False)
    
    # Create pie chart
    fig = px.pie(
        waste_composition, 
        values='weight_kg', 
        names='waste_type',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'><h3>Tren Volume Sampah per Jenis</h3>", unsafe_allow_html=True)
    
    # Prepare data for trend analysis
    # Fix the resample error by converting the date to datetime if needed
    daily_waste = df_filtered.groupby(['date', 'waste_type'])['weight_kg'].sum().reset_index()
    
    # Convert to proper datetime if needed
    daily_waste['date'] = pd.to_datetime(daily_waste['date'])
    
    # Create a pivot table for easier plotting
    waste_pivot = daily_waste.pivot(index='date', columns='waste_type', values='weight_kg').fillna(0)
    
    # Group by week without resample (to avoid errors)
    waste_by_week = []
    
    # Get the list of unique weeks
    weekly_dates = sorted(list(set([d.isocalendar()[1] for d in daily_waste['date']])))
    
    for week in weekly_dates:
        # Filter for this week
        week_data = daily_waste[daily_waste['date'].apply(lambda x: x.isocalendar()[1] == week)]
        
        # Group by waste type and sum
        for waste_type in waste_types:
            week_sum = week_data[week_data['waste_type'] == waste_type]['weight_kg'].sum()
            # Use the first date in this week
            first_date = week_data['date'].min()
            
            waste_by_week.append({
                'date': first_date,
                'week': week,
                'waste_type': waste_type,
                'weight_kg': week_sum
            })
    
    weekly_waste_df = pd.DataFrame(waste_by_week)
    
    # Plot the trend
    fig = px.line(
        weekly_waste_df, 
        x='date', 
        y='weight_kg', 
        color='waste_type',
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Berat (kg)",
        legend_title="Jenis Sampah",
        margin=dict(t=0, b=0, l=0, r=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row2_col2:
    # Community Engagement
    st.markdown("<div class='card'><h3>Community Engagement</h3>", unsafe_allow_html=True)
    
    # Leaderboard
    st.subheader("🏆 Leaderboard Kecamatan")
    top_kecamatan = df_filtered.groupby('kecamatan')['weight_kg'].sum().reset_index()
    top_kecamatan = top_kecamatan.sort_values('weight_kg', ascending=False).head(5)
    
    for i, (_, row) in enumerate(top_kecamatan.iterrows()):
        st.markdown(f"**{i+1}. {row['kecamatan']}** - {row['weight_kg']:,.2f} kg")
    
    # User growth
    st.subheader("📈 Pertumbuhan Pengguna")
    
    fig = px.line(
        users_df,
        x='month',
        y='users',
        markers=True
    )
    fig.update_layout(
        xaxis_title="Bulan",
        yaxis_title="Jumlah Pengguna",
        margin=dict(t=0, b=0, l=0, r=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    total_users = users_df.iloc[-1]['users']
    monthly_growth = (users_df.iloc[-1]['users'] / users_df.iloc[-2]['users'] - 1) * 100
    
    st.metric("Total Pengguna", f"{total_users:,}", f"{monthly_growth:.1f}% dari bulan lalu")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ROW 3: Success Stories and Prediction
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.markdown("<div class='card'><h3>Success Stories</h3>", unsafe_allow_html=True)
    
    # Simulated success stories
    stories = [
        {
            "title": "Bank Sampah Berseri",
            "content": "Kelompok Bank Sampah Berseri berhasil mengolah sampah plastik menjadi tas dan aksesori yang kini dipasarkan secara online. Pendapatan dari penjualan meningkat 200% dalam 6 bulan terakhir.",
            "testimonial": "Sistem pengelolaan sampah OASYS membantu kami mendapatkan pasokan bahan baku yang konsisten dan berkualitas untuk produk kami.",
            "author": "Ibu Siti, Koordinator Bank Sampah Berseri"
        },
        {
            "title": "Kompos Makmur",
            "content": "Komunitas pertanian urban 'Kompos Makmur' menggunakan pupuk organik dari sampah organik yang dikumpulkan. Hasil panen sayuran mereka meningkat hingga 30% sejak menggunakan pupuk dari sampah organik.",
            "testimonial": "Kualitas pupuk dari sampah organik yang terkelola dengan baik sangat membantu kami menghasilkan sayuran organik berkualitas tinggi.",
            "author": "Pak Joko, Ketua Komunitas Kompos Makmur"
        }
    ]
    
    for story in stories:
        st.markdown(f"""
        <div class="success-story">
            <h4>{story['title']}</h4>
            <p>{story['content']}</p>
            <div class="testimonial">
                "{story['testimonial']}"
                <br>
                <b>- {story['author']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


with row3_col2:
    st.markdown("<div class='card'><h3>Prediksi Volume Sampah</h3>", unsafe_allow_html=True)
    
    # Prepare data for prediction visualization
    dates = df['date'].sort_values().unique()
    daily_totals = df.groupby('date')['weight_kg'].sum().reset_index()
    
    # Get the last date from historical data
    last_date = dates[-1]  # Using the date directly
    
    # Create future dates for prediction
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    
    # Create a simple trend-based prediction
    avg_daily = daily_totals['weight_kg'].mean()
    std_daily = daily_totals['weight_kg'].std()
    
    # Trend factor
    trend = 0.002  # Slight upward trend
    future_predictions = []
    
    for i, date in enumerate(future_dates):
        # Add some seasonality
        day_of_week = date.weekday()
        seasonal_factor = 1.0
        if day_of_week >= 5:  # Weekend
            seasonal_factor = 1.2
        predicted_value = avg_daily * (1 + trend * i) * seasonal_factor
        future_predictions.append({
            'date': date,
            'weight_kg': predicted_value,
            'type': 'Prediksi'
        })
    
    # Convert historical data
    historical_data = [{'date': row['date'], 'weight_kg': row['weight_kg'], 'type': 'Historis'}
                      for _, row in daily_totals.iterrows()]
    
    # Combine datasets
    prediction_df = pd.DataFrame(historical_data + future_predictions)
    
    # Make sure date column is datetime type
    prediction_df['date'] = pd.to_datetime(prediction_df['date'])
    
    # Plot
    fig = px.line(
        prediction_df,
        x='date',
        y='weight_kg',
        color='type',
        title='Proyeksi Volume Sampah 30 Hari ke Depan',
        labels={'date': 'Tanggal', 'weight_kg': 'Volume (kg)', 'type': 'Data'},
        color_discrete_map={'Historis': '#1f77b4', 'Prediksi': '#ff7f0e'}
    )
    
    # Improve layout
    fig.update_layout(
        legend_title_text='',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_type='date'
    )
    
    # Add a vertical line using shapes instead of add_vline
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=last_date,
                y0=0,
                x1=last_date,
                y1=1,
                line=dict(
                    color="gray",
                    width=1,
                    dash="dash",
                )
            )
        ],
        annotations=[
            dict(
                x=last_date,
                y=1,
                xref="x",
                yref="paper",
                text="Mulai Prediksi",
                showarrow=False,
                xanchor="right"
            )
        ]
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Add some explanation
    st.markdown("""
    <div style='font-size:0.9em;color:#666;'>
    * Prediksi menggunakan tren historis dengan peningkatan 0.2% per hari
    * Volume sampah pada akhir pekan cenderung 20% lebih tinggi
    </div>
    """, unsafe_allow_html=True)
    
    # Close the card div
    st.markdown("</div>", unsafe_allow_html=True)

# ROW 4: Tips and Data Table
row4_col1, row4_col2 = st.columns([1, 2])

with row4_col1:
    st.markdown("<div class='card'><h3>Tips Pengelolaan Sampah</h3>", unsafe_allow_html=True)
    
    # Sampah of the week
    st.markdown("""
    <div class="tips-card">
        <h4>📦 Sampah Minggu Ini: Botol Plastik</h4>
        <p><b>Tahukah kamu?</b> Satu botol plastik membutuhkan 450 tahun untuk terurai di alam.</p>
        <p><b>Tips:</b></p>
        <ul>
            <li>Cuci dan keringkan sebelum membuang</li>
            <li>Pisahkan tutup botol (berbeda jenis plastik)</li>
            <li>Tekan botol untuk menghemat ruang</li>
            <li>Gunakan kembali untuk pot tanaman atau kerajinan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # General tips
    st.markdown("""
    <div class="tips-card">
        <h4>💡 Tips Umum</h4>
        <p><b>Cara Memilah Sampah di Rumah:</b></p>
        <ul>
            <li><span style="color:green"><b>Hijau</b></span> - Sampah organik (sisa makanan, daun)</li>
            <li><span style="color:blue"><b>Biru</b></span> - Sampah anorganik (plastik, kertas, logam)</li>
            <li><span style="color:red"><b>Merah</b></span> - Sampah B3 (baterai, lampu, obat)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with row4_col2:
    st.markdown("<div class='card'><h3>Data Pengambilan Sampah</h3>", unsafe_allow_html=True)

    # ✅ Convert ke datetime dulu (fix utama)
    display_df = df_filtered.copy()
    display_df['date'] = pd.to_datetime(display_df['date'], errors='coerce')

    # ✅ Drop baris yang gagal dikonversi
    display_df = display_df.dropna(subset=['date'])

    # ✅ Format tanggal
    display_df['date'] = display_df['date'].dt.strftime('%d-%m-%Y')

    # Ambil kolom yang diperlukan
    display_df = display_df[['date', 'time', 'kecamatan', 'waste_type', 'weight_kg', 'status']]
    display_df.columns = ['Tanggal', 'Waktu', 'Kecamatan', 'Jenis Sampah', 'Berat (kg)', 'Status']

    # Tampilkan tabel
    st.dataframe(display_df.sort_values(by='Tanggal', ascending=False), use_container_width=True)
