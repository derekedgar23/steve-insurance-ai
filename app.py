import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    .chat-container {
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .steve-response {
        background-color: var(--primary-background);
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: var(--text-color);
    }
    .user-message {
        background-color: var(--secondary-background);
        border-left: 4px solid #00aa44;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: var(--text-color);
    }
    
    /* Light mode colors */
    [data-theme="light"] {
        --text-color: #1f4e79;
        --background-color: #f8f9fa;
        --border-color: #dee2e6;
        --primary-background: #e3f2fd;
        --secondary-background: #f1f8e9;
    }
    
    /* Dark mode colors */
    [data-theme="dark"] {
        --text-color: #87ceeb;
        --background-color: #2d3748;
        --border-color: #4a5568;
        --primary-background: #2c5282;
        --secondary-background: #2f855a;
    }
    
    /* Auto-detect system theme */
    @media (prefers-color-scheme: light) {
        :root {
            --text-color: #1f4e79;
            --background-color: #f8f9fa;
            --border-color: #dee2e6;
            --primary-background: #e3f2fd;
            --secondary-background: #f1f8e9;
        }
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --text-color: #87ceeb;
            --background-color: #2d3748;
            --border-color: #4a5568;
            --primary-background: #2c5282;
            --secondary-background: #2f855a;
        }
    }
</style>
""", unsafe_allow_html=True)

class SteveWebApp:
    def __init__(self):
        self.insurance_keywords = {
            'claims': ['claim', 'loss', 'incident', 'accident', 'damage', 'injury', 'liability'],
            'policy': ['policy', 'premium', 'coverage', 'deductible', 'limit', 'insured'],
            'financial': ['reserve', 'paid', 'outstanding', 'incurred', 'expense', 'settlement'],
            'legal': ['litigation', 'suit', 'attorney', 'court', 'judgment'],
            'regulatory': ['filing', 'compliance', 'audit', 'examination']
        }
        
        if 'analyzed_data' not in st.session_state:
            st.session_state.analyzed_data = {}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def analyze_csv(self, uploaded_file):
        try:
            df = pd.read_csv(uploaded_file)
            filename = uploaded_file.name
            
            analysis = {
                'filename': filename,
                'type': 'CSV',
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dataframe': df,
                'insurance_columns': [],
                'data_quality': {},
                'recommendations': [],
                'summary_stats': {}
            }
            
            for col in df.columns:
                col_lower = col.lower()
                for category, keywords in self.insurance_keywords.items():
                    if any(keyword in col_lower for keyword in keywords):
                        analysis['insurance_columns'].append({
                            'column': col,
                            'category': category
                        })
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if not df[col].empty and not df[col].isna().all():
                    analysis['summary_stats'][col] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'sum': float(df[col].sum()),
                        'std': float(df[col].std())
                    }
            
            missing_data = {}
            for col in df.columns:
                null_pct = (df[col].isnull().sum() / len(df)) * 100
                if null_pct > 5:
                    missing_data[col] = {
                        'count': int(df[col].isnull().sum()),
                        'percentage': round(null_pct, 1)
                    }
            
            analysis['data_quality']['missing_data'] = missing_data
            analysis['data_quality']['duplicates'] = len(df) - len(df.drop_duplicates())
            
            if analysis['insurance_columns']:
                analysis['recommendations'].append("📊 Insurance data detected - ready for analysis!")
            if missing_data:
                analysis['recommendations'].append("⚠️ Some columns have missing data - consider data cleaning")
            if analysis['data_quality']['duplicates'] > 0:
                analysis['recommendations'].append(f"🔄 Found {analysis['data_quality']['duplicates']} duplicate rows")
            
            st.session_state.analyzed_data[filename] = analysis
            return analysis
            
        except Exception as e:
            return {'filename': uploaded_file.name, 'error': str(e)}
    
    def analyze_excel(self, uploaded_file):
        try:
            filename = uploaded_file.name
            uploaded_file.seek(0)
            
            try:
                excel_data = pd.ExcelFile(uploaded_file)
            except Exception as e:
                return {'filename': filename, 'error': f"Could not read Excel file: {str(e)}"}
            
            sheets_analysis = {}
            total_rows = 0
            total_insurance_cols = 0
            
            for sheet_name in excel_data.sheet_names:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
                    
                    if df.empty:
                        continue
                    
                    sheet_analysis = {
                        'sheet_name': sheet_name,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': list(df.columns),
                        'dataframe': df,
                        'insurance_columns': [],
                        'summary_stats': {},
                        'data_quality': {}
                    }
                    
                    for col in df.columns:
                        col_lower = str(col).lower()
                        for category, keywords in self.insurance_keywords.items():
                            if any(keyword in col_lower for keyword in keywords):
                                sheet_analysis['insurance_columns'].append({
                                    'column': col,
                                    'category': category
                                })
                    
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if not df[col].empty and not df[col].isna().all():
                            sheet_analysis['summary_stats'][col] = {
                                'mean': float(df[col].mean()),
                                'sum': float(df[col].sum()),
                                'min': float(df[col].min()),
                                'max': float(df[col].max()),
                                'median': float(df[col].median())
                            }
                    
                    missing_data = {}
                    for col in df.columns:
                        null_pct = (df[col].isnull().sum() / len(df)) * 100
                        if null_pct > 5:
                            missing_data[col] = {
                                'count': int(df[col].isnull().sum()),
                                'percentage': round(null_pct, 1)
                            }
                    
                    sheet_analysis['data_quality']['missing_data'] = missing_data
                    sheet_analysis['data_quality']['duplicates'] = len(df) - len(df.drop_duplicates())
                    
                    sheets_analysis[sheet_name] = sheet_analysis
                    total_rows += len(df)
                    total_insurance_cols += len(sheet_analysis['insurance_columns'])
                    
                except Exception as e:
                    sheets_analysis[sheet_name] = {
                        'sheet_name': sheet_name,
                        'error': f"Error processing sheet: {str(e)}"
                    }
            
            analysis = {
                'filename': filename,
                'type': 'Excel',
                'sheets': sheets_analysis,
                'total_sheets': len(excel_data.sheet_names),
                'processed_sheets': len([s for s in sheets_analysis.values() if 'error' not in s]),
                'total_rows': total_rows,
                'total_insurance_cols': total_insurance_cols,
                'recommendations': []
            }
            
            if total_insurance_cols > 0:
                analysis['recommendations'].append("📊 Insurance data detected across Excel sheets - ready for analysis!")
            if analysis['processed_sheets'] < analysis['total_sheets']:
                analysis['recommendations'].append("⚠️ Some sheets couldn't be processed - check for formatting issues")
            
            st.session_state.analyzed_data[filename] = analysis
            return analysis
            
        except Exception as e:
            return {'filename': uploaded_file.name, 'error': f"Excel analysis failed: {str(e)}"}

    
    def analyze_pdf(self, uploaded_file):
        try:
            filename = uploaded_file.name
            text_content = ""
            
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            except:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
            
            analysis = {
                'filename': filename,
                'type': 'PDF',
                'text_content': text_content[:5000],
                'document_type': self._identify_document_type(text_content),
                'key_findings': self._extract_key_findings(text_content),
                'insurance_metrics': self._extract_insurance_metrics(text_content)
            }
            
            st.session_state.analyzed_data[filename] = analysis
            return analysis
            
        except Exception as e:
            return {'filename': uploaded_file.name, 'error': str(e)}
    
    def _identify_document_type(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ['claim', 'loss run', 'incident']):
            return "Claims Document"
        elif any(word in text_lower for word in ['policy', 'certificate', 'coverage']):
            return "Policy Document"
        elif any(word in text_lower for word in ['financial', 'balance sheet', 'income']):
            return "Financial Document"
        else:
            return "General Document"
    
    def _extract_key_findings(self, text):
        findings = []
        
        money_pattern = r'\$[\d,]+\.?\d*'
        money_matches = re.findall(money_pattern, text)
        if money_matches:
            findings.append(f"Found {len(money_matches)} monetary amounts")
        
        date_pattern = r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'
        date_matches = re.findall(date_pattern, text)
        if date_matches:
            findings.append(f"Found {len(date_matches)} date references")
        
        return findings
    
    def _extract_insurance_metrics(self, text):
        metrics = {}
        
        money_pattern = r'\$[\d,]+\.?\d*'
        money_amounts = re.findall(money_pattern, text)
        if money_amounts:
            metrics["monetary_amounts"] = len(money_amounts)
        
        limit_pattern = r'limit[s]?\s*[:\-]?\s*\$[\d,]+\.?\d*'
        limit_matches = re.findall(limit_pattern, text, re.IGNORECASE)
        if limit_matches:
            metrics["coverage_limits"] = len(limit_matches)
        
        return metrics
    
    def generate_chat_response(self, question):
        if not st.session_state.analyzed_data:
            return "🤖 Hi! I don't have any documents analyzed yet. Please upload and analyze some insurance files first, then I can answer questions about them!"
        
        question_lower = question.lower()
        response_parts = []
        
        if any(greeting in question_lower for greeting in ['hello', 'hi', 'hey']):
            files_count = len(st.session_state.analyzed_data)
            files_list = list(st.session_state.analyzed_data.keys())
            return f"🤖 Hello! I'm STEVE, your insurance analysis AI. I've analyzed {files_count} file(s): {', '.join(files_list)}. What would you like to know about your data?"
        
        if 'column' in question_lower or 'field' in question_lower:
            for filename, analysis in st.session_state.analyzed_data.items():
                if analysis['type'] in ['CSV', 'Excel']:
                    if analysis['type'] == 'CSV':
                        cols = analysis['column_names']
                        insurance_cols = [col['column'] for col in analysis['insurance_columns']]
                        response_parts.append(f"📊 **{filename}** has {len(cols)} columns:")
                        response_parts.append(f"• All columns: {', '.join(cols)}")
                        if insurance_cols:
                            response_parts.append(f"• Insurance columns: {', '.join(insurance_cols)}")
                    elif analysis['type'] == 'Excel':
                        for sheet_name, sheet_info in analysis['sheets'].items():
                            cols = sheet_info['column_names']
                            response_parts.append(f"📋 **{filename} - {sheet_name}**: {', '.join(cols)}")
        
        elif any(word in question_lower for word in ['summary', 'overview', 'about']):
            for filename, analysis in st.session_state.analyzed_data.items():
                if analysis['type'] == 'CSV':
                    response_parts.append(f"📄 **{filename}** Summary:")
                    response_parts.append(f"• {analysis['rows']:,} rows × {analysis['columns']} columns")
                    response_parts.append(f"• Insurance columns detected: {len(analysis['insurance_columns'])}")
                elif analysis['type'] == 'Excel':
                    response_parts.append(f"📄 **{filename}** Summary:")
                    response_parts.append(f"• {analysis['total_sheets']} sheets, {analysis.get('total_rows', 0):,} total rows")
                    response_parts.append(f"• {analysis.get('total_insurance_cols', 0)} insurance columns across all sheets")
                    response_parts.append(f"• Successfully processed: {analysis.get('processed_sheets', 0)}/{analysis['total_sheets']} sheets")
                elif analysis['type'] == 'PDF':
                    response_parts.append(f"📄 **{filename}** Summary:")
                    response_parts.append(f"• Document Type: {analysis['document_type']}")
                    response_parts.append(f"• Key Findings: {', '.join(analysis['key_findings'])}")
        
        elif any(word in question_lower for word in ['total', 'sum', 'amount']):
            found_totals = False
            for filename, analysis in st.session_state.analyzed_data.items():
                if analysis['type'] == 'CSV' and analysis.get('summary_stats'):
                    response_parts.append(f"💰 **{filename}** - Financial Totals:")
                    for col, stats in analysis['summary_stats'].items():
                        if stats['sum'] > 0 and any(keyword in col.lower() for keyword in ['premium', 'amount', 'paid', 'loss', 'reserve']):
                            response_parts.append(f"• {col}: ${stats['sum']:,.2f} (avg: ${stats['mean']:,.2f})")
                            found_totals = True
                elif analysis['type'] == 'Excel':
                    response_parts.append(f"💰 **{filename}** - Excel Financial Totals:")
                    for sheet_name, sheet_info in analysis['sheets'].items():
                        if 'error' not in sheet_info and sheet_info.get('summary_stats'):
                            for col, stats in sheet_info['summary_stats'].items():
                                if stats['sum'] > 0 and any(keyword in col.lower() for keyword in ['premium', 'amount', 'paid', 'loss', 'reserve']):
                                    response_parts.append(f"• {sheet_name} - {col}: ${stats['sum']:,.2f}")
                                    found_totals = True
            
            if not found_totals:
                response_parts.append("🔍 I couldn't find obvious monetary columns. Could you specify which amounts you're interested in?")
        
        elif any(word in question_lower for word in ['quality', 'missing', 'problem', 'issue']):
            for filename, analysis in st.session_state.analyzed_data.items():
                if analysis['type'] == 'CSV':
                    response_parts.append(f"⚠️ **{filename}** - Data Quality Report:")
                    
                    if analysis['data_quality']['missing_data']:
                        response_parts.append("• Missing data issues:")
                        for col, info in list(analysis['data_quality']['missing_data'].items())[:3]:
                            response_parts.append(f"  - {col}: {info['percentage']}% missing ({info['count']} rows)")
                    else:
                        response_parts.append("• ✅ No significant missing data")
                    
                    if analysis['data_quality']['duplicates'] > 0:
                        response_parts.append(f"• ⚠️ {analysis['data_quality']['duplicates']} duplicate rows found")
                    else:
                        response_parts.append("• ✅ No duplicate rows")
        
        if not response_parts:
            available_files = ', '.join(st.session_state.analyzed_data.keys())
            response_parts.append(f"🤔 I'm not sure how to answer that. I have data from: {available_files}")
            response_parts.append("\nTry asking:")
            response_parts.append("• 'What columns do I have?'")
            response_parts.append("• 'Show me a summary'")
            response_parts.append("• 'What are the total amounts?'")
            response_parts.append("• 'Any data quality issues?'")
        
        return "\n".join(response_parts)

def main():
    steve = SteveWebApp()
    
    st.markdown('<div class="main-header">🤖 STEVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Smart Technology for Evaluating insurance documents and Validating Exposures</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose insurance files to analyze",
            type=['csv', 'xlsx', 'xls', 'pdf'],
            accept_multiple_files=True,
            help="Upload CSV, Excel, or PDF files for analysis"
        )
        
        if uploaded_files:
            st.write(f"📄 {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🚀 Analyze Files", type="primary"):
                with st.spinner("STEVE is analyzing your files..."):
                    for file in uploaded_files:
                        if file.name.endswith('.csv'):
                            result = steve.analyze_csv(file)
                        elif file.name.endswith(('.xlsx', '.xls')):
                            result = steve.analyze_excel(file)
                        elif file.name.endswith('.pdf'):
                            result = steve.analyze_pdf(file)
                        
                        if 'error' not in result:
                            st.success(f"✅ Analyzed: {file.name}")
                        else:
                            st.error(f"❌ Error with {file.name}: {result['error']}")
                
                st.rerun()
    
    if not st.session_state.analyzed_data:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 🚀 Welcome to STEVE!
            
            Your AI-powered insurance document analysis platform specifically designed for P&C insurance companies.
            
            **What STEVE can analyze:**
            - 📊 Claims data and loss runs
            - 📋 Policy information and coverage details
            - 💰 Financial reports and premium data
            - 📄 Regulatory filings and compliance documents
            
            **Get started:**
            1. Upload your files using the sidebar
            2. Click "Analyze Files"
            3. Chat with STEVE about your data
            
            Ready to revolutionize your insurance data analysis? Upload your files to begin! 🚀
            """)
    
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "💬 Chat with STEVE", "📈 Visualizations"])
        
        with tab1:
            st.header("📊 Analysis Results")
            
            for filename, analysis in st.session_state.analyzed_data.items():
                with st.expander(f"📄 {filename}", expanded=True):
                    if 'error' in analysis:
                        st.error(f"Error: {analysis['error']}")
                        continue
                    
                    if analysis['type'] == 'CSV':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Rows", f"{analysis['rows']:,}")
                        with col2:
                            st.metric("Columns", analysis['columns'])
                        with col3:
                            st.metric("Insurance Columns", len(analysis['insurance_columns']))
                        with col4:
                            quality_score = "Good" if not analysis['data_quality']['missing_data'] else "Issues"
                            st.metric("Data Quality", quality_score)
                        
                        if analysis['insurance_columns']:
                            st.subheader("🏢 Insurance Columns Detected")
                            for col_info in analysis['insurance_columns']:
                                st.write(f"• **{col_info['column']}** ({col_info['category']})")
                        
                        if analysis['data_quality']['missing_data']:
                            st.subheader("⚠️ Data Quality Issues")
                            for col, info in analysis['data_quality']['missing_data'].items():
                                st.warning(f"**{col}**: {info['percentage']}% missing data ({info['count']} rows)")
                    
                    elif analysis['type'] == 'Excel':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Sheets", analysis['total_sheets'])
                        with col2:
                            st.metric("Processed Sheets", f"{analysis.get('processed_sheets', 0)}")
                        with col3:
                            st.metric("Total Rows", f"{analysis.get('total_rows', 0):,}")
                        with col4:
                            st.metric("Insurance Columns", analysis.get('total_insurance_cols', 0))
                        
                        for sheet_name, sheet_info in analysis['sheets'].items():
                            if 'error' in sheet_info:
                                st.error(f"**{sheet_name}**: {sheet_info['error']}")
                            else:
                                st.write(f"📋 **{sheet_name}**: {sheet_info['rows']:,} rows × {sheet_info['columns']} columns")
                                
                                if sheet_info.get('insurance_columns'):
                                    st.write("🏢 Insurance columns detected:")
                                    for col_info in sheet_info['insurance_columns']:
                                        st.write(f"  • **{col_info['column']}** ({col_info['category']})")
                                
                                if sheet_info.get('data_quality', {}).get('missing_data'):
                                    st.write("⚠️ Data quality issues:")
                                    for col, info in sheet_info['data_quality']['missing_data'].items():
                                        st.warning(f"  {col}: {info['percentage']}% missing")
                        
                        if analysis.get('recommendations'):
                            st.subheader("💡 Recommendations")
                            for rec in analysis['recommendations']:
                                st.write(f"• {rec}")

                    
                    elif analysis['type'] == 'PDF':
                        st.write(f"**Document Type**: {analysis['document_type']}")
                        if analysis['key_findings']:
                            st.write("**Key Findings**:")
                            for finding in analysis['key_findings']:
                                st.write(f"• {finding}")
        
        with tab2:
            st.header("💬 Chat with STEVE")
            
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            for i, chat in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f'<div class="user-message"><strong>You:</strong> {chat["user"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="steve-response"><strong>🤖 STEVE:</strong> {chat["steve"]}</div>', unsafe_allow_html=True)
            
            user_question = st.text_input("Ask STEVE about your insurance data:", placeholder="What columns do I have?", key="chat_input")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Send 🚀"):
                    if user_question:
                        response = steve.generate_chat_response(user_question)
                        st.session_state.chat_history.append({
                            "user": user_question,
                            "steve": response
                        })
                        st.rerun()
            
            with col2:
                if st.button("Clear Chat 🗑️"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            st.subheader("💡 Suggested Questions")
            suggested_questions = [
                "What columns do I have?",
                "Show me a summary of my data",
                "What are the total premium amounts?",
                "Are there any data quality issues?",
                "Tell me about claims data",
                "What do you recommend?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(suggested_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"suggest_{i}"):
                        response = steve.generate_chat_response(question)
                        st.session_state.chat_history.append({
                            "user": question,
                            "steve": response
                        })
                        st.rerun()
        
        with tab3:
            st.header("📈 Data Visualizations")
            
            for filename, analysis in st.session_state.analyzed_data.items():
                if analysis['type'] == 'CSV' and analysis.get('summary_stats'):
                    st.subheader(f"📊 {filename}")
                    
                    monetary_cols = []
                    for col, stats in analysis['summary_stats'].items():
                        if any(keyword in col.lower() for keyword in ['premium', 'amount', 'paid', 'loss', 'reserve']):
                            monetary_cols.append(col)
                    
                    if monetary_cols and len(monetary_cols) > 0:
                        df = analysis['dataframe']
                        
                        col = monetary_cols[0]
                        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        stats_data = []
                        for col in monetary_cols:
                            if col in analysis['summary_stats']:
                                stats = analysis['summary_stats'][col]
                                stats_data.append({
                                    'Column': col,
                                    'Total': f"${stats['sum']:,.2f}",
                                    'Average': f"${stats['mean']:,.2f}",
                                    'Median': f"${stats['median']:,.2f}",
                                    'Min': f"${stats['min']:,.2f}",
                                    'Max': f"${stats['max']:,.2f}"
                                })
                        
                        if stats_data:
                            st.subheader("📋 Financial Summary")
                            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

if __name__ == "__main__":
    main()
