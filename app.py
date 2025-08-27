import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Risk Assessment",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 35px;
        text-align: center;
        color: #00d4ff;
        margin-bottom: 30px;
    }
    .sub-header {
        color: #ffffff;
        font-size: 24px;
        margin-bottom: 15px;
    }
    .info-text {
        font-size: 18px;
        color: #e0e0e0;
        margin-bottom: 10px;
    }
    .center-text {
        text-align: center;
        color: #ffffff;
        font-size: 18px;
        margin-bottom: 8px;
    }
    .metric-container {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #262730;
        border-radius: 10px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance with caching"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_returns(data):
    """Calculate log returns"""
    returns = np.log(data['Close'] / data['Close'].shift(1))
    return returns.dropna()

def calculate_var_historical(returns, confidence_level):
    """Calculate VaR using Historical Simulation"""
    if len(returns) == 0:
        return 0
    var_value = -np.percentile(returns.dropna(), (1 - confidence_level) * 100)
    return var_value

def calculate_var_vcov(returns, confidence_level):
    """Calculate VaR using Variance-Covariance method"""
    if len(returns) == 0:
        return 0
    mean_return = returns.mean()
    std_dev = returns.std()
    var_value = -(mean_return + norm.ppf(confidence_level) * std_dev)
    return var_value

def calculate_var_monte_carlo(returns, confidence_level, n_simulations=1000):
    """Calculate VaR using Monte Carlo Simulation"""
    if len(returns) == 0:
        return 0
    mean_return = returns.mean()
    std_dev = returns.std()
    
    mc_results = []
    for _ in range(n_simulations):
        simulated_returns = np.random.normal(mean_return, std_dev, len(returns))
        var_sim = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        mc_results.append(var_sim)
    
    var_value = -np.mean(mc_results)
    return var_value

def calculate_var_ecf(returns, confidence_level):
    """Calculate VaR using Empirical Cumulative Function"""
    if len(returns) == 0:
        return 0
    var_value = -np.percentile(returns.dropna(), (1 - confidence_level) * 100)
    return var_value

def create_technical_chart(data, symbol):
    """Create technical analysis chart"""
    # Calculate technical indicators
    data = data.copy()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_5'] = data['Close'].ewm(span=5).mean()
    
    # Calculate Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    data['BB_std'] = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
    data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        subplot_titles=(f'{symbol} Price Chart', 'Volume'),
                        row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Price'), row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], 
                            mode='lines', name='SMA 20', 
                            line=dict(color='orange', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], 
                            mode='lines', name='SMA 50', 
                            line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_5'], 
                            mode='lines', name='EMA 5', 
                            line=dict(color='yellow', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], 
                            mode='lines', name='BB Upper', 
                            line=dict(color='gray', dash='dash'), 
                            opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], 
                            mode='lines', name='BB Lower', 
                            line=dict(color='gray', dash='dash'), 
                            opacity=0.5, fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                        marker_color=colors, opacity=0.7), row=2, col=1)
    
    fig.update_layout(height=700, 
                      template='plotly_dark',
                      title=f'{symbol} Technical Analysis',
                      xaxis_rangeslider_visible=False)
    
    return fig

def create_return_distribution_plot(returns):
    """Create return distribution histogram with normal curve overlay"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set dark theme
    plt.style.use('dark_background')
    
    # Histogram
    n, bins, patches = ax.hist(returns, bins=30, density=True, alpha=0.7, 
                              color='#0d6efd', label='Actual Returns', edgecolor='white', linewidth=0.5)
    
    # Normal distribution overlay
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_curve = norm.pdf(x, mu, sigma)
    ax.plot(x, normal_curve, color='#ffc107', linestyle='--', linewidth=3, label='Normal Distribution')
    
    # Add VaR line for 95% confidence
    var_95 = -np.percentile(returns, 5)
    ax.axvline(-var_95, color='red', linestyle='-', linewidth=2, label=f'VaR 95% = {var_95:.4f}')
    
    ax.set_xlabel('Returns (Log Return)', color='white', fontsize=12)
    ax.set_ylabel('Density', color='white', fontsize=12)
    ax.set_title('Stock Return Distribution', color='white', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    return fig

def display_var_results(returns, confidence_level, investment_amount, var_method):
    """Display VaR calculation results"""
    conf_level = confidence_level / 100
    
    # Calculate VaR based on selected method
    if var_method == "Historical Simulation":
        var_value = calculate_var_historical(returns, conf_level)
        method_name = "Historical Simulation"
        method_description = "Non-parametric method using historical data percentiles"
    elif var_method == "Variance-Covariance":
        var_value = calculate_var_vcov(returns, conf_level)
        method_name = "Variance-Covariance"
        method_description = "Parametric method assuming normal distribution"
    elif var_method == "Monte Carlo":
        var_value = calculate_var_monte_carlo(returns, conf_level)
        method_name = "Monte Carlo Simulation"
        method_description = "Simulation-based approach with 1000 iterations"
    else:  # ECF
        var_value = calculate_var_ecf(returns, conf_level)
        method_name = "Empirical Cumulative Function"
        method_description = "Empirical percentile-based method"
    
    loss_simulation = investment_amount * var_value
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📉 VaR Value", 
            value=f"{var_value:.6f}",
            help="Value at Risk as a proportion of investment"
        )
    
    with col2:
        st.metric(
            label="💰 Potential Loss", 
            value=f"${abs(loss_simulation):,.2f}",
            delta=f"{(var_value*100):.2f}%",
            help="Maximum expected loss in dollar terms"
        )
    
    with col3:
        st.metric(
            label="🎯 Confidence Level", 
            value=f"{confidence_level}%",
            help="Statistical confidence in the VaR estimate"
        )
    
    # Method information
    st.info(f"**Method:** {method_name} - {method_description}")
    
    # Risk interpretation
    if var_value < 0.02:
        risk_level = "🟢 Low Risk"
        risk_color = "green"
    elif var_value < 0.05:
        risk_level = "🟡 Medium Risk"
        risk_color = "orange"
    else:
        risk_level = "🔴 High Risk"
        risk_color = "red"
    
    st.markdown(f"**Risk Assessment:** <span style='color:{risk_color}; font-weight:bold;'>{risk_level}</span>", 
                unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">KOMPUTASI STATISTIKA LANJUT</h1>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["👤 Pembuat", "💡 Insight", "📊 Analisis"])
    
    with tab1:
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2 style='color: #00d4ff; margin-bottom: 30px;'>STOCK RISK ASSESSMENT WITH VALUE AT RISK (VAR)</h2>
            <br>
            <h3 style='color: white; margin-bottom: 10px;'>RAKA RIZKY RAMADHAN</h3>
            <h4 style='color: #cccccc; margin-bottom: 10px;'>24050122130100</h4>
            <h4 style='color: #cccccc; margin-bottom: 30px;'>KOMPUTASI STATISTIKA LANJUT - B</h4>
            <br>
            <h4 style='color: white; margin-bottom: 10px;'>DEPARTEMEN STATISTIKA</h4>
            <h4 style='color: white; margin-bottom: 10px;'>FAKULTAS SAINS DAN MATEMATIKA</h4>
            <h4 style='color: white; margin-bottom: 10px;'>UNIVERSITAS DIPONEGORO</h4>
            <h4 style='color: #00d4ff; margin-bottom: 10px;'>2025</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<h2 class="sub-header">📈 Pengertian Saham</h2>', unsafe_allow_html=True)
            st.markdown('''
            <div class="info-text">
            Saham merupakan surat berharga yang menunjukkan bagian kepemilikan atas suatu perusahaan. 
            Pemilik saham berhak atas dividen serta potensi keuntungan dari kenaikan harga saham.
            
            <br><br>
            
            <strong>Karakteristik Saham:</strong><br>
            • Memberikan hak kepemilikan perusahaan<br>
            • Potensi return tinggi dengan risiko tinggi<br>
            • Likuiditas tinggi di pasar modal<br>
            • Nilai dipengaruhi kondisi ekonomi dan kinerja perusahaan
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('<h2 class="sub-header">🎯 Pengertian Value at Risk (VaR)</h2>', unsafe_allow_html=True)
            st.markdown('''
            <div class="info-text">
            Value at Risk (VaR) adalah ukuran risiko yang digunakan untuk memperkirakan potensi kerugian 
            maksimum dari suatu investasi dalam periode tertentu dengan tingkat kepercayaan tertentu. 
            
            <br><br>
            
            <strong>Formula Umum VaR:</strong><br>
            VaR = μ + σ × Z<sub>α</sub><br><br>
            
            Dimana:<br>
            • μ = Mean return<br>
            • σ = Standard deviation<br>
            • Z<sub>α</sub> = Critical value pada confidence level α
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h2 class="sub-header">🔧 Kapan Menggunakan Masing-masing Metode VaR</h2>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-text">
            <strong>🔹 Historical Simulation:</strong><br>
            • Data historis mencukupi<br>
            • Pendekatan non-parametrik<br>
            • Tidak memerlukan asumsi distribusi<br>
            • Cocok untuk data dengan pola tidak normal<br><br>
            
            <strong>🔹 Variance-Covariance:</strong><br>
            • Asumsi distribusi normal<br>
            • Perhitungan cepat dan efisien<br>
            • Cocok untuk portofolio sederhana<br>
            • Memerlukan stabilitas volatilitas<br><br>
            
            <strong>🔹 Monte Carlo Simulation:</strong><br>
            • Skenario kompleks dan multiple<br>
            • Asumsi distribusi normal multivariat<br>
            • Fleksibel untuk berbagai kondisi<br>
            • Akurat untuk portofolio besar<br><br>
            
            <strong>🔹 ECF (Empirical Cumulative Function):</strong><br>
            • Data return dengan skewness dan kurtosis ekstrim<br>
            • Distribusi empiris dari data historis<br>
            • Tidak memerlukan asumsi distribusi parametrik<br>
            • Cocok untuk data dengan fat tails
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<h2 class="sub-header">⚠️ Catatan Penting</h2>', unsafe_allow_html=True)
            st.markdown('''
            <div class="info-text">
            • Pastikan koneksi internet stabil untuk mengakses data Yahoo Finance<br>
            • Gunakan simbol saham yang benar (contoh: AAPL, MSFT, GOOGL)<br>
            • Untuk saham Indonesia, tambahkan .JK (contoh: BBCA.JK)<br>
            • VaR adalah estimasi, bukan jaminan kerugian maksimum<br>
            • Selalu pertimbangkan faktor fundamental dalam investasi
            </div>
            ''', unsafe_allow_html=True)
    
    with tab3:
        # Sidebar for inputs
        with st.sidebar:
            st.markdown('<h2 style="color: #00d4ff;">🎛️ Input Parameters</h2>', unsafe_allow_html=True)
            
            stock_symbol = st.text_input(
                "📊 Stock Symbol:", 
                value="AAPL", 
                placeholder="e.g., AAPL, MSFT, GOOGL, BBCA.JK",
                help="Enter stock symbol. For Indonesian stocks, add .JK (e.g., BBCA.JK)"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "📅 Start Date:", 
                    value=pd.Timestamp.now() - pd.DateOffset(years=1),
                    help="Select start date for historical data"
                )
            with col2:
                end_date = st.date_input(
                    "📅 End Date:", 
                    value=pd.Timestamp.now(),
                    help="Select end date for historical data"
                )
            
            var_method = st.selectbox(
                "🔢 VaR Method:",
                ["Historical Simulation", "Variance-Covariance", "Monte Carlo", "ECF"],
                help="Choose the VaR calculation method"
            )
            
            confidence_level = st.slider(
                "🎯 Confidence Level (%):", 
                min_value=90, max_value=99, value=95,
                help="Statistical confidence level for VaR calculation"
            )
            
            investment_amount = st.number_input(
                "💰 Investment Amount:", 
                value=1000, min_value=1, step=100,
                help="Enter your investment amount in USD"
            )
            
            run_analysis = st.button("📈 Analyze Stock", type="primary", use_container_width=True)
        
        # Main analysis area
        if run_analysis and stock_symbol:
            # Validate date range
            if start_date >= end_date:
                st.error("❌ Start date must be before end date!")
                return
            
            # Fetch stock data
            with st.spinner(f"Fetching data for {stock_symbol.upper()}..."):
                stock_data = get_stock_data(stock_symbol.upper(), start_date, end_date)
            
            if stock_data is not None and not stock_data.empty:
                # Calculate returns
                returns = calculate_returns(stock_data)
                
                if len(returns) < 30:
                    st.warning("⚠️ Limited data available. Results may not be reliable. Consider extending the date range.")
                
                # Display basic stock info
                st.success(f"✅ Successfully loaded {len(stock_data)} days of data for {stock_symbol.upper()}")
                
                # Create tabs for results
                var_tab, chart_tab, stats_tab = st.tabs(["📊 VaR Analysis", "📈 Technical Chart", "📋 Statistics"])
                
                with var_tab:
                    st.markdown('<h2 class="sub-header">Value at Risk Analysis</h2>', unsafe_allow_html=True)
                    
                    # Display VaR results
                    display_var_results(returns, confidence_level, investment_amount, var_method)
                    
                    # Return distribution plot
                    st.markdown('<h3 class="sub-header">Return Distribution</h3>', unsafe_allow_html=True)
                    dist_plot = create_return_distribution_plot(returns)
                    st.pyplot(dist_plot)
                
                with chart_tab:
                    st.markdown('<h2 class="sub-header">Technical Analysis</h2>', unsafe_allow_html=True)
                    
                    # Technical chart
                    tech_chart = create_technical_chart(stock_data.copy(), stock_symbol.upper())
                    st.plotly_chart(tech_chart, use_container_width=True)
                    
                    # Current price info
                    current_price = stock_data['Close'].iloc[-1]
                    price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
                    price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                    with col3:
                        st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,.0f}")
                
                with stats_tab:
                    st.markdown('<h2 class="sub-header">Statistical Summary</h2>', unsafe_allow_html=True)
                    
                    # Calculate comprehensive statistics
                    mean_ret = returns.mean()
                    var_ret = returns.var()
                    std_ret = returns.std()
                    skew_ret = stats.skew(returns.dropna())
                    kurt_ret = stats.kurtosis(returns.dropna())
                    min_ret = returns.min()
                    max_ret = returns.max()
                    
                    # Create two columns for statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Return Statistics**")
                        stats_df1 = pd.DataFrame({
                            'Metric': ['Mean Return', 'Standard Deviation', 'Variance', 'Minimum Return'],
                            'Value': [f"{mean_ret:.6f}", f"{std_ret:.6f}", f"{var_ret:.6f}", f"{min_ret:.6f}"]
                        })
                        st.dataframe(stats_df1, hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.markdown("**📈 Distribution Statistics**")
                        stats_df2 = pd.DataFrame({
                            'Metric': ['Maximum Return', 'Skewness', 'Kurtosis', 'Observations'],
                            'Value': [f"{max_ret:.6f}", f"{skew_ret:.6f}", f"{kurt_ret:.6f}", f"{len(returns)}"]
                        })
                        st.dataframe(stats_df2, hide_index=True, use_container_width=True)
                    
                    # Risk metrics
                    st.markdown("**⚠️ Risk Metrics**")
                    annualized_vol = std_ret * np.sqrt(252)  # 252 trading days
                    sharpe_ratio = (mean_ret * 252) / annualized_vol if annualized_vol != 0 else 0
                    
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    with risk_col1:
                        st.metric("Annualized Volatility", f"{annualized_vol:.4f}")
                    with risk_col2:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.4f}")
                    with risk_col3:
                        st.metric("Daily VaR (95%)", f"{calculate_var_historical(returns, 0.95):.6f}")
                
            else:
                st.error(f"❌ Could not fetch data for {stock_symbol.upper()}. Please check:")
                st.markdown("""
                - Stock symbol is correct
                - Internet connection is stable
                - Try different date range
                - For Indonesian stocks, add .JK (e.g., BBCA.JK)
                """)
        
        elif run_analysis:
            st.error("❌ Please enter a stock symbol!")

if __name__ == "__main__":
    main()