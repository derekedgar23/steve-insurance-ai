import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any, Tuple
warnings.filterwarnings('ignore')

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(
    page_title="STEVE - Insurance AI",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: var(--text-color);
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: var(--text-color);
        opacity: 0.8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-card {
        background-color: var(--primary-background);
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #856404;
    }
    .success-card {
        background-color: #d1edff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #004085;
    }
    
    @media (prefers-color-scheme: light) {
        :root {
            --text-color: #1f4e79;
            --primary-background: #e3f2fd;
        }
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #87ceeb;
            --primary-background: #2c5282;
        }
    }
</style>
""", unsafe_allow_html=True)

class AdvancedInsuranceAnalyzer:
    def __init__(self):
        self.insurance_patterns = {
            'claims': {
                'keywords': ['claim', 'loss', 'incident', 'accident', 'damage', 'injury', 'liability', 'settlement'],
                'amount_cols': ['paid', 'incurred', 'reserve', 'settlement', 'indemnity', 'expense'],
                'date_cols': ['loss_date', 'report_date', 'close_date', 'occurrence', 'reported'],
                'status_cols': ['status', 'state', 'closed', 'open', 'pending']
            },
            'policy': {
                'keywords': ['policy', 'premium', 'coverage', 'deductible', 'limit', 'insured', 'renewal'],
                'amount_cols': ['premium', 'limit', 'deductible', 'coverage', 'written', 'earned'],
                'date_cols': ['effective', 'expiration', 'inception', 'renewal'],
                'identifier_cols': ['policy_number', 'account', 'producer', 'agent']
            },
            'financial': {
                'keywords': ['reserve', 'paid', 'outstanding', 'incurred', 'expense', 'alae', 'ulae'],
                'ratio_analysis': True,
                'trend_analysis': True
            }
        }
        
        if 'analyzed_data' not in st.session_state:
            st.session_state.analyzed_data = {}
        if 'analysis_insights' not in st.session_state:
            st.session_state.analysis_insights = {}
    
    def detect_data_type(self, df: pd.DataFrame) -> str:
        column_text = ' '.join(df.columns.astype(str)).lower()
        
        claims_score = sum(1 for keyword in self.insurance_patterns['claims']['keywords'] if keyword in column_text)
        policy_score = sum(1 for keyword in self.insurance_patterns['policy']['keywords'] if keyword in column_text)
        
        if claims_score > policy_score:
            return 'claims_data'
        elif policy_score > claims_score:
            return 'policy_data'
        else:
            return 'mixed_data'
    
    def detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        date_columns = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            if any(date_word in col_lower for date_word in ['date', 'time', 'effective', 'expiration', 'loss', 'report']):
                try:
                    sample_data = df[col].dropna().head(10)
                    if not sample_data.empty:
                        pd.to_datetime(sample_data, errors='raise')
                        date_columns.append(col)
                except:
                    pass
            
            elif df[col].dtype == 'object':
                try:
                    sample_data = df[col].dropna().head(5)
                    if not sample_data.empty:
                        parsed = pd.to_datetime(sample_data, errors='coerce')
                        if parsed.notna().sum() >= 3:
                            date_columns.append(col)
                except:
                    pass
        
        return date_columns
    
    def detect_monetary_columns(self, df: pd.DataFrame) -> List[Dict]:
        monetary_columns = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            is_monetary = any(keyword in col_lower for keyword in 
                            ['premium', 'amount', 'paid', 'reserve', 'limit', 'deductible', 
                             'incurred', 'settlement', 'expense', 'cost', 'value', 'indemnity'])
            
            if (is_monetary or df[col].dtype in ['int64', 'float64']) and df[col].dtype != 'object':
                if not df[col].dropna().empty:
                    sample_values = df[col].dropna().head(10)
                    if sample_values.min() >= 0 and sample_values.max() > 100:
                        
                        category = 'unknown'
                        if any(word in col_lower for word in ['premium', 'written', 'earned']):
                            category = 'premium'
                        elif any(word in col_lower for word in ['paid', 'payment']):
                            category = 'paid_loss'
                        elif any(word in col_lower for word in ['reserve', 'outstanding']):
                            category = 'reserve'
                        elif any(word in col_lower for word in ['incurred', 'total']):
                            category = 'incurred_loss'
                        elif any(word in col_lower for word in ['limit', 'coverage']):
                            category = 'limit'
                        elif any(word in col_lower for word in ['deductible']):
                            category = 'deductible'
                        
                        monetary_columns.append({
                            'column': col,
                            'category': category,
                            'total': float(df[col].sum()),
                            'mean': float(df[col].mean()),
                            'median': float(df[col].median()),
                            'max': float(df[col].max()),
                            'min': float(df[col].min())
                        })
        
        return monetary_columns
    
    def calculate_insurance_ratios(self, df: pd.DataFrame, monetary_cols: List[Dict]) -> Dict:
        ratios = {}
        
        premium_cols = [col for col in monetary_cols if col['category'] == 'premium']
        paid_loss_cols = [col for col in monetary_cols if col['category'] == 'paid_loss']
        incurred_cols = [col for col in monetary_cols if col['category'] in ['incurred_loss', 'paid_loss']]
        reserve_cols = [col for col in monetary_cols if col['category'] == 'reserve']
        
        if premium_cols and incurred_cols:
            premium_total = sum(col['total'] for col in premium_cols)
            incurred_total = sum(col['total'] for col in incurred_cols)
            
            if premium_total > 0:
                ratios['loss_ratio'] = (incurred_total / premium_total) * 100
                ratios['loss_ratio_interpretation'] = self.interpret_loss_ratio(ratios['loss_ratio'])
        
        if paid_loss_cols and reserve_cols:
            paid_total = sum(col['total'] for col in paid_loss_cols)
            reserve_total = sum(col['total'] for col in reserve_cols)
            
            if paid_total > 0:
                ratios['reserve_to_paid_ratio'] = (reserve_total / paid_total) * 100
                ratios['reserve_adequacy'] = self.interpret_reserve_ratio(ratios['reserve_to_paid_ratio'])
        
        if 'claims_data' in str(df.columns).lower():
            total_claims = len(df)
            if premium_cols:
                avg_premium = sum(col['mean'] for col in premium_cols)
                if avg_premium > 0:
                    ratios['frequency'] = total_claims / len(df) if len(df) > 0 else 0
        
        return ratios
    
    def interpret_loss_ratio(self, loss_ratio: float) -> str:
        if loss_ratio < 60:
            return "Excellent - Very profitable"
        elif loss_ratio < 80:
            return "Good - Profitable"
        elif loss_ratio < 100:
            return "Acceptable - Breaking even"
        elif loss_ratio < 120:
            return "Concerning - Unprofitable"
        else:
            return "Poor - Significant losses"
    
    def interpret_reserve_ratio(self, reserve_ratio: float) -> str:
        if reserve_ratio < 20:
            return "Low reserves - May be under-reserved"
        elif reserve_ratio < 50:
            return "Moderate reserves - Typical range"
        elif reserve_ratio < 100:
            return "High reserves - Conservative approach"
        else:
            return "Very high reserves - May be over-reserved"
    
    def analyze_trends(self, df: pd.DataFrame, date_cols: List[str], monetary_cols: List[Dict]) -> Dict:
        trends = {}
        
        if not date_cols or not monetary_cols:
            return trends
        
        date_col = date_cols[0]
        
        try:
            df_trend = df.copy()
            df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
            df_trend = df_trend.dropna(subset=[date_col])
            
            if len(df_trend) < 2:
                return trends
            
            df_trend['period'] = df_trend[date_col].dt.to_period('M')
            
            for mon_col in monetary_cols[:3]:
                col_name = mon_col['column']
                if col_name in df_trend.columns:
                    monthly_data = df_trend.groupby('period')[col_name].agg(['sum', 'count', 'mean']).reset_index()
                    
                    if len(monthly_data) >= 3:
                        x = np.arange(len(monthly_data))
                        y = monthly_data['sum'].values
                        slope = np.polyfit(x, y, 1)[0]
                        r_value = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                        
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        trend_strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
                        
                        trends[col_name] = {
                            'direction': trend_direction,
                            'strength': trend_strength,
                            'correlation': r_value,
                            'monthly_avg': monthly_data['sum'].mean(),
                            'latest_month': monthly_data['sum'].iloc[-1],
                            'change_rate': slope
                        }
        
        except Exception as e:
            trends['error'] = f"Trend analysis failed: {str(e)}"
        
        return trends
    
    def detect_anomalies(self, df: pd.DataFrame, monetary_cols: List[Dict]) -> List[Dict]:
        anomalies = []
        
        for mon_col in monetary_cols:
            col_name = mon_col['column']
            if col_name in df.columns:
                data = df[col_name].dropna()
                
                if len(data) > 10:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    
                    if len(outliers) > 0:
                        anomalies.append({
                            'column': col_name,
                            'outlier_count': len(outliers),
                            'outlier_percentage': (len(outliers) / len(data)) * 100,
                            'max_outlier': float(outliers.max()),
                            'min_outlier': float(outliers.min()),
                            'severity': 'high' if len(outliers) / len(data) > 0.1 else 'moderate'
                        })
        
        return anomalies
    
    def generate_insights(self, analysis: Dict) -> List[str]:
        insights = []
        
        if analysis.get('data_quality', {}).get('missing_data'):
            missing_count = len(analysis['data_quality']['missing_data'])
            insights.append(f"🔍 Data Quality: {missing_count} columns have significant missing data - prioritize data collection improvements")
        
        ratios = analysis.get('ratios', {})
        if 'loss_ratio' in ratios:
            lr = ratios['loss_ratio']
            insights.append(f"📊 Loss Ratio: {lr:.1f}% - {ratios['loss_ratio_interpretation']}")
            
            if lr > 100:
                insights.append(f"⚠️ Action Required: Loss ratio above 100% indicates unprofitability - review pricing and underwriting")
        
        if 'reserve_to_paid_ratio' in ratios:
            insights.append(f"💰 Reserve Analysis: {ratios['reserve_adequacy']}")
        
        trends = analysis.get('trends', {})
        for col, trend_data in trends.items():
            if isinstance(trend_data, dict) and 'direction' in trend_data:
                direction = trend_data['direction']
                strength = trend_data['strength']
                insights.append(f"📈 Trend Alert: {col} shows {strength} {direction} trend over time")
        
        anomalies = analysis.get('anomalies', [])
        for anomaly in anomalies:
            if anomaly['severity'] == 'high':
                insights.append(f"🚨 Anomaly Detected: {anomaly['column']} has {anomaly['outlier_percentage']:.1f}% outliers - investigate unusual claims")
        
        if analysis.get('rows', 0) > 0:
            row_count = analysis['rows']
            if row_count < 100:
                insights.append(f"📊 Sample Size: Only {row_count} records - consider larger dataset for more reliable analysis")
            elif row_count > 10000:
                insights.append(f"📊 Large Dataset: {row_count:,} records provide excellent statistical power")
        
        return insights
    
    def analyze_csv(self, uploaded_file):
        try:
            df = pd.read_csv(uploaded_file)
            filename = uploaded_file.name
            
            data_type = self.detect_data_type(df)
            date_columns = self.detect_date_columns(df)
            monetary_columns = self.detect_monetary_columns(df)
            
            ratios = self.calculate_insurance_ratios(df, monetary_columns)
            trends = self.analyze_trends(df, date_columns, monetary_columns)
            anomalies = self.detect_anomalies(df, monetary_columns)
            
            missing_data = {}
            for col in df.columns:
                null_pct = (df[col].isnull().sum() / len(df)) * 100
                if null_pct > 5:
                    missing_data[col] = {
                        'count': int(df[col].isnull().sum()),
                        'percentage': round(null_pct, 1)
                    }
            
            analysis = {
                'filename': filename,
                'type': 'CSV',
                'data_type': data_type,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dataframe': df,
                'date_columns': date_columns,
                'monetary_columns': monetary_columns,
                'ratios': ratios,
                'trends': trends,
                'anomalies': anomalies,
                'data_quality': {
                    'missing_data': missing_data,
                    'duplicates': len(df) - len(df.drop_duplicates())
                }
            }
            
            insights = self.generate_insights(analysis)
            analysis['insights'] = insights
            
            st.session_state.analyzed_data[filename] = analysis
            return analysis
            
        except Exception as e:
            return {'filename': uploaded_file.name, 'error': str(e)}
    
    def create_advanced_visualizations(self, analysis: Dict) -> List[go.Figure]:
        figures = []
        
        if analysis['type'] != 'CSV' or 'dataframe' not in analysis:
            return figures
        
        df = analysis['dataframe']
        monetary_cols = analysis['monetary_columns']
        
        if len(monetary_cols) >= 2:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Premium Distribution', 'Loss Distribution', 'Claims by Amount', 'Ratio Analysis'),
                specs=[[{"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            premium_col = next((col for col in monetary_cols if col['category'] == 'premium'), monetary_cols[0])
            fig.add_trace(go.Histogram(x=df[premium_col['column']], name='Premium', nbinsx=20), row=1, col=1)
            
            loss_col = next((col for col in monetary_cols if col['category'] in ['paid_loss', 'incurred_loss']), monetary_cols[1])
            fig.add_trace(go.Histogram(x=df[loss_col['column']], name='Losses', nbinsx=20), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=df[premium_col['column']], 
                y=df[loss_col['column']],
                mode='markers',
                name='Premium vs Loss',
                marker=dict(color='blue', opacity=0.6)
            ), row=2, col=1)
            
            ratios = analysis.get('ratios', {})
            if ratios:
                ratio_names = list(ratios.keys())
                ratio_values = [ratios[key] for key in ratio_names if isinstance(ratios[key], (int, float))]
                
                if ratio_values:
                    fig.add_trace(go.Bar(
                        x=ratio_names[:len(ratio_values)],
                        y=ratio_values,
                        name='Key Ratios'
                    ), row=2, col=2)
            
            fig.update_layout(height=700, title_text="Insurance Data Analysis Dashboard")
            figures.append(fig)
        
        if analysis['date_columns'] and len(monetary_cols) > 0:
            date_col = analysis['date_columns'][0]
            
            try:
                df_trend = df.copy()
                df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors='coerce')
                df_trend = df_trend.dropna(subset=[date_col])
                
                if len(df_trend) > 1:
                    df_trend = df_trend.sort_values(date_col)
                    df_trend['month'] = df_trend[date_col].dt.to_period('M').astype(str)
                    
                    fig_trend = go.Figure()
                    
                    for mon_col in monetary_cols[:3]:
                        monthly_sum = df_trend.groupby('month')[mon_col['column']].sum()
                        fig_trend.add_trace(go.Scatter(
                            x=monthly_sum.index,
                            y=monthly_sum.values,
                            mode='lines+markers',
                            name=f"{mon_col['column']} ({mon_col['category']})"
                        ))
                    
                    fig_trend.update_layout(
                        title="Temporal Trends Analysis",
                        xaxis_title="Month",
                        yaxis_title="Amount ($)",
                        height=400
                    )
                    figures.append(fig_trend)
            except:
                pass
        
        return figures

def main():
    analyzer = AdvancedInsuranceAnalyzer()
    
    st.markdown('<div class="main-header">🤖 STEVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Insurance Analytics Platform</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose insurance files to analyze",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload CSV or Excel files for advanced analysis"
        )
        
        if uploaded_files:
            st.write(f"📄 {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🚀 Analyze Files", type="primary"):
                with st.spinner("STEVE is performing advanced analysis..."):
                    for file in uploaded_files:
                        if file.name.endswith('.csv'):
                            result = analyzer.analyze_csv(file)
                        
                        if 'error' not in result:
                            st.success(f"✅ Analyzed: {file.name}")
                        else:
                            st.error(f"❌ Error with {file.name}: {result['error']}")
                
                st.rerun()
    
    if not st.session_state.analyzed_data:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 🚀 Welcome to Advanced STEVE!
            
            **🎯 Advanced Insurance Analytics:**
            - **Loss Ratio Analysis** - Profitability assessment
            - **Reserve Adequacy** - Financial health metrics
            - **Trend Detection** - Temporal pattern analysis
            - **Anomaly Detection** - Unusual claims identification
            - **Data Quality Assessment** - Completeness evaluation
            
            **📊 Sophisticated Visualizations:**
            - Interactive dashboards
            - Trend analysis charts  
            - Statistical distributions
            - Comparative analytics
            
            **🔍 Actionable Insights:**
            - Business recommendations
            - Risk assessments
            - Performance benchmarks
            - Strategic guidance
            
            Upload your insurance data to experience enterprise-grade analytics! 🚀
            """)
    
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Advanced Analysis", "📈 Visualizations", "🎯 Key Insights", "💬 Chat"])
        
        with tab1:
            st.header("📊 Advanced Insurance Analysis")
            
            for filename, analysis in st.session_state.analyzed_data.items():
                with st.expander(f"📄 {filename}", expanded=True):
                    if 'error' in analysis:
                        st.error(f"Error: {analysis['error']}")
                        continue
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Records", f"{analysis['rows']:,}")
                    with col2:
                        st.metric("Data Type", analysis['data_type'].replace('_', ' ').title())
                    with col3:
                        st.metric("Monetary Columns", len(analysis['monetary_columns']))
                    with col4:
                        st.metric("Date Columns", len(analysis['date_columns']))
                    
                    if analysis.get('ratios'):
                        st.subheader("💰 Financial Analysis")
                        ratios = analysis['ratios']
                        
                        if 'loss_ratio' in ratios:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Loss Ratio", f"{ratios['loss_ratio']:.1f}%")
                            with col2:
                                interpretation = ratios.get('loss_ratio_interpretation', '')
                                color = "success" if "Excellent" in interpretation or "Good" in interpretation else "warning"
                                st.markdown(f"<div class='{color}-card'>{interpretation}</div>", unsafe_allow_html=True)
                        
                        if 'reserve_to_paid_ratio' in ratios:
                            st.markdown(f"**Reserve Adequacy:** {ratios['reserve_adequacy']}")
                    
                    if analysis['monetary_columns']:
                        st.subheader("💵 Monetary Analysis")
                        for mon_col in analysis['monetary_columns']:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(f"{mon_col['column']}", f"${mon_col['total']:,.0f}")
                            with col2:
                                st.metric("Average", f"${mon_col['mean']:,.0f}")
                            with col3:
                                st.metric("Median", f"${mon_col['median']:,.0f}")
                            with col4:
                                st.metric("Max", f"${mon_col['max']:,.0f}")
                    
                    if analysis.get('anomalies'):
                        st.subheader("🚨 Anomaly Detection")
                        for anomaly in analysis['anomalies']:
                            severity_color = "error" if anomaly['severity'] == 'high' else "warning"
                            st.markdown(f"""
                            <div class='warning-card'>
                            <strong>{anomaly['column']}</strong>: {anomaly['outlier_count']} outliers 
                            ({anomaly['outlier_percentage']:.1f}% of data) - Severity: {anomaly['severity']}
                            </div>
                            """, unsafe_allow_html=True)
        
        with tab2:
            st.header("📈 Advanced Visualizations")
            
            for filename, analysis in st.session_state.analyzed_data.items():
                if 'error' not in analysis:
                    st.subheader(f"📊 {filename}")
                    
                    figures = analyzer.create_advanced_visualizations(analysis)
                    for fig in figures:
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("🎯 Key Insights & Recommendations")
            
            for filename, analysis in st.session_state.analyzed_data.items():
                if 'error' not in analysis and 'insights' in analysis:
                    st.subheader(f"💡 {filename}")
                    
                    for insight in analysis['insights']:
                        if '🚨' in insight or '⚠️' in insight:
                            st.markdown(f'<div class="warning-card">{insight}</div>', unsafe_allow_html=True)
                        elif '✅' in insight or 'Excellent' in insight:
                            st.markdown(f'<div class="success-card">{insight}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        
        with tab4:
            st.header("💬 Chat with STEVE")
            st.info("🤖 Advanced AI chat coming soon! For now, explore the detailed analysis in other tabs.")

if __name__ == "__main__":
    main()
