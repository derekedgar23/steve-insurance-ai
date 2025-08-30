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
import json
import openai
from typing import Dict, List, Any
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

class SteveWithLLM:
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
        if 'llm_provider' not in st.session_state:
            st.session_state.llm_provider = 'openai'
    
    def setup_llm_client(self, provider: str, api_key: str):
        """Setup the LLM client based on provider"""
        try:
            if provider == 'openai':
                openai.api_key = api_key
                return True
            elif provider == 'anthropic':
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                return True
            elif provider == 'groq':
                from groq import Groq
                self.groq_client = Groq(api_key=api_key)
                return True
            return False
        except Exception as e:
            st.error(f"Error setting up {provider}: {str(e)}")
            return False
    
    def generate_llm_response(self, user_question: str, context_data: str, provider: str) -> str:
        """Generate response using the selected LLM provider"""
        
        system_prompt = f"""You are STEVE (Smart Technology for Evaluating insurance documents and Validating Exposures), an AI assistant specialized in Property & Casualty insurance analysis.

You have access to the following analyzed insurance data:
{context_data}

Your role is to:
1. Answer questions about insurance data with expertise
2. Provide insights on claims, policies, premiums, and financial data
3. Identify trends, patterns, and potential issues
4. Give actionable recommendations for P&C insurance operations
5. Explain complex insurance concepts clearly

Always be professional, accurate, and focus on practical insurance applications. Use specific data from the context when possible."""

        try:
            if provider == 'openai':
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_question}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                return response.choices[0].message.content
                
            elif provider == 'anthropic':
                message = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_question}]
                )
                return message.content[0].text
                
            elif provider == 'groq':
                completion = self.groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_question}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                return completion.choices[0].message.content
                
            elif provider == 'huggingface':
                return self.generate_huggingface_response(user_question, context_data)
                
        except Exception as e:
            return f"Sorry, I encountered an error with the {provider} API: {str(e)}. Please check your API key and try again."
    
    def generate_huggingface_response(self, user_question: str, context_data: str) -> str:
        """Generate response using Hugging Face transformers (local/free option)"""
        try:
            from transformers import pipeline
            
            if 'hf_generator' not in st.session_state:
                st.session_state.hf_generator = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    return_full_text=False,
                    max_new_tokens=200
                )
            
            prompt = f"As an insurance AI assistant analyzing: {context_data[:500]}...\n\nUser question: {user_question}\n\nResponse:"
            
            response = st.session_state.hf_generator(prompt)[0]['generated_text']
            return f"🤖 Based on your insurance data: {response}"
            
        except Exception as e:
            return "I'm having trouble with the local AI model. Please try one of the API options instead."
    
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
                        'summary_stats': {}
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
                                'max': float(df[col].max())
                            }
                    
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
                'total_rows': total_rows,
                'total_insurance_cols': total_insurance_cols
            }
            
            st.session_state.analyzed_data[filename] = analysis
            return analysis
            
        except Exception as e:
            return {'filename': uploaded_file.name, 'error': f"Excel analysis failed: {str(e)}"}
    
    def prepare_context_for_llm(self) -> str:
        """Prepare analyzed data context for the LLM"""
        if not st.session_state.analyzed_data:
            return "No insurance data has been analyzed yet."
        
        context_parts = []
        
        for filename, analysis in st.session_state.analyzed_data.items():
            if analysis['type'] == 'CSV':
                context_parts.append(f"CSV File: {filename}")
                context_parts.append(f"- {analysis['rows']:,} rows, {analysis['columns']} columns")
                context_parts.append(f"- Columns: {', '.join(analysis['column_names'])}")
                
                if analysis['insurance_columns']:
                    ins_cols = [f"{col['column']} ({col['category']})" for col in analysis['insurance_columns']]
                    context_parts.append(f"- Insurance columns: {', '.join(ins_cols)}")
                
                if analysis['summary_stats']:
                    context_parts.append("- Financial data:")
                    for col, stats in analysis['summary_stats'].items():
                        if any(kw in col.lower() for kw in ['premium', 'amount', 'paid', 'loss', 'reserve']):
                            context_parts.append(f"  * {col}: Total ${stats['sum']:,.2f}, Average ${stats['mean']:,.2f}")
                
            elif analysis['type'] == 'Excel':
                context_parts.append(f"Excel File: {filename}")
                context_parts.append(f"- {analysis['total_sheets']} sheets, {analysis['total_rows']:,} total rows")
                
                for sheet_name, sheet_info in analysis['sheets'].items():
                    if 'error' not in sheet_info:
                        context_parts.append(f"- Sheet '{sheet_name}': {sheet_info['rows']:,} rows")
                        if sheet_info['insurance_columns']:
                            ins_cols = [col['column'] for col in sheet_info['insurance_columns']]
                            context_parts.append(f"  * Insurance columns: {', '.join(ins_cols)}")
        
        return "\n".join(context_parts)
    
    def generate_chat_response(self, question: str, provider: str) -> str:
        """Generate intelligent chat response using LLM"""
        if not st.session_state.analyzed_data:
            return "🤖 Hi! I don't have any documents analyzed yet. Please upload and analyze some insurance files first, then I can answer questions about them using advanced AI!"
        
        context_data = self.prepare_context_for_llm()
        
        if provider == 'fallback':
            return self.generate_fallback_response(question)
        
        return self.generate_llm_response(question, context_data, provider)
    
    def generate_fallback_response(self, question: str) -> str:
        """Simple fallback response without LLM API"""
        question_lower = question.lower()
        
        if 'hello' in question_lower or 'hi' in question_lower:
            return "🤖 Hello! I'm STEVE, your insurance AI assistant. I can analyze your data and answer questions. For advanced AI responses, please configure an API key in the sidebar!"
        
        context = self.prepare_context_for_llm()
        return f"🤖 Here's what I found in your data:\n\n{context}\n\nFor more intelligent analysis and insights, please set up an LLM API in the sidebar settings."

def main():
    steve = SteveWithLLM()
    
    st.markdown('<div class="main-header">🤖 STEVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Smart Technology for Evaluating insurance documents and Validating Exposures</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("🔧 LLM Configuration")
        
        provider = st.selectbox(
            "Choose LLM Provider:",
            ['openai', 'anthropic', 'groq', 'huggingface', 'fallback'],
            help="Select your preferred AI model provider"
        )
        
        if provider != 'fallback' and provider != 'huggingface':
            api_key = st.text_input(
                f"Enter {provider.upper()} API Key:",
                type="password",
                help=f"Get your API key from {provider}.com"
            )
            
            if api_key:
                if steve.setup_llm_client(provider, api_key):
                    st.success(f"✅ Connected to {provider.upper()}")
                    st.session_state.llm_provider = provider
                else:
                    st.error(f"❌ Failed to connect to {provider.upper()}")
            else:
                st.warning(f"⚠️ Please enter your {provider.upper()} API key")
        elif provider == 'huggingface':
            st.info("🤗 Using local Hugging Face model (free, but slower)")
            st.session_state.llm_provider = provider
        else:
            st.info("📊 Using basic pattern matching (no LLM)")
            st.session_state.llm_provider = provider
        
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
                        
                        if 'error' not in result:
                            st.success(f"✅ Analyzed: {file.name}")
                        else:
                            st.error(f"❌ Error with {file.name}: {result['error']}")
                
                st.rerun()
    
    if not st.session_state.analyzed_data:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 🚀 Welcome to STEVE with AI!
            
            Your AI-powered insurance document analysis platform with advanced language model integration.
            
            **🧠 AI-Powered Features:**
            - **GPT-4** integration for advanced analysis
            - **Claude** support for detailed insights
            - **Groq** for fast responses
            - **Local AI** options available
            
            **📊 What STEVE can analyze:**
            - Claims data and loss runs
            - Policy information and coverage details
            - Financial reports and premium data
            - Regulatory filings and compliance documents
            
            **🔧 Setup:**
            1. Choose your AI provider in the sidebar
            2. Enter your API key (or use free local option)
            3. Upload your insurance files
            4. Chat with STEVE using advanced AI!
            
            Ready for intelligent insurance analysis? Configure your AI and upload files! 🚀
            """)
    
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "💬 Chat with AI STEVE", "📈 Visualizations"])
        
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
                    
                    elif analysis['type'] == 'Excel':
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Sheets", analysis['total_sheets'])
                        with col2:
                            st.metric("Total Rows", f"{analysis['total_rows']:,}")
                        with col3:
                            st.metric("Insurance Columns", analysis['total_insurance_cols'])
                        
                        for sheet_name, sheet_info in analysis['sheets'].items():
                            if 'error' not in sheet_info:
                                st.write(f"📋 **{sheet_name}**: {sheet_info['rows']:,} rows × {sheet_info['columns']} columns")
        
        with tab2:
            st.header("💬 Chat with AI STEVE")
            
            provider_status = st.session_state.get('llm_provider', 'fallback')
            if provider_status == 'fallback':
                st.warning("⚠️ Using basic responses. Configure an LLM API in the sidebar for intelligent AI chat!")
            elif provider_status == 'huggingface':
                st.info("🤗 Using local Hugging Face AI model")
            else:
                st.success(f"🤖 Connected to {provider_status.upper()} AI")
            
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="user-message"><strong>You:</strong> {chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="steve-response"><strong>🤖 AI STEVE:</strong> {chat["steve"]}</div>', unsafe_allow_html=True)
            
            user_question = st.text_input("Ask STEVE anything about your insurance data:", placeholder="What insights can you provide about my claims data?", key="chat_input")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Send 🚀"):
                    if user_question:
                        with st.spinner("🤖 AI STEVE is thinking..."):
                            response = steve.generate_chat_response(user_question, st.session_state.llm_provider)
                            st.session_state.chat_history.append({
                                "user": user_question,
                                "steve": response
                            })
                        st.rerun()
            
            with col2:
                if st.button("Clear Chat 🗑️"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            st.subheader("💡 AI-Powered Questions to Try")
            ai_questions = [
                "What trends do you see in my insurance data?",
                "Are there any anomalies or red flags I should investigate?",
                "What recommendations do you have for improving our loss ratios?",
                "Can you explain the relationship between premiums and claims in my data?",
                "What insights can you provide about our policy portfolio?",
                "How does our claims frequency compare to industry standards?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(ai_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"ai_suggest_{i}"):
                        with st.spinner("🤖 AI STEVE is analyzing..."):
                            response = steve.generate_chat_response(question, st.session_state.llm_provider)
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

if __name__ == "__main__":
    main()
