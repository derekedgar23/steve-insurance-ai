import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
from typing import Dict, List, Any
warnings.filterwarnings('ignore')

# AI Model Integration Options
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class SteveAITrainer:
    def __init__(self):
        # Insurance domain knowledge base for training
        self.insurance_knowledge = {
            "loss_ratio_benchmarks": {
                "auto": {"excellent": 60, "good": 75, "acceptable": 85, "poor": 100},
                "property": {"excellent": 55, "good": 70, "acceptable": 80, "poor": 95},
                "workers_comp": {"excellent": 65, "good": 80, "acceptable": 90, "poor": 105},
                "general_liability": {"excellent": 50, "good": 65, "acceptable": 75, "poor": 85}
            },
            "reserve_patterns": {
                "auto": {"typical_reserve_ratio": 0.3, "development_period": 24},
                "property": {"typical_reserve_ratio": 0.15, "development_period": 12},
                "workers_comp": {"typical_reserve_ratio": 0.8, "development_period": 60}
            },
            "industry_insights": [
                "Loss ratios above 100% indicate unprofitability and require immediate attention",
                "Increasing claim frequency often indicates underwriting issues or market changes",
                "Reserve adequacy is critical for financial stability and regulatory compliance",
                "Seasonal patterns in claims data often reveal important business insights",
                "Large claims require individual attention and may indicate systemic issues"
            ]
        }
        
        # Training data templates for insurance analysis
        self.training_examples = [
            {
                "input": "Loss ratio is 95% for auto insurance",
                "output": "This 95% loss ratio for auto insurance is concerning and approaches unprofitability. Industry benchmark for acceptable auto loss ratio is 85%. Recommend reviewing pricing strategy and underwriting guidelines."
            },
            {
                "input": "Claims frequency increased 15% year over year",
                "output": "A 15% increase in claims frequency is significant and warrants investigation. This could indicate: 1) Deteriorating underwriting standards, 2) Changes in risk profile, 3) Economic factors affecting claims behavior, or 4) Fraud trends. Recommend detailed analysis by coverage type and geography."
            },
            {
                "input": "Reserve to paid ratio is 0.8 for workers compensation",
                "output": "A reserve to paid ratio of 0.8 (80%) for workers compensation is within normal range, as workers comp typically has longer development periods and higher reserve needs due to medical inflation and wage replacement. Monitor for adequacy and consider actuarial review."
            }
        ]

    def create_insurance_fine_tuning_dataset(self, analyzed_data: Dict) -> List[Dict]:
        """Create training dataset from analyzed insurance data"""
        training_data = []
        
        for filename, analysis in analyzed_data.items():
            if 'error' in analysis:
                continue
            
            # Create training examples from analysis
            context = self.build_context_string(analysis)
            
            # Generate question-answer pairs
            qa_pairs = [
                {
                    "messages": [
                        {"role": "system", "content": "You are STEVE, an expert P&C insurance analyst. Provide detailed, actionable insights based on insurance data analysis."},
                        {"role": "user", "content": f"Analyze this insurance data: {context}"},
                        {"role": "assistant", "content": self.generate_expert_analysis(analysis)}
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": "You are STEVE, an expert P&C insurance analyst."},
                        {"role": "user", "content": f"What are the key risks in this data: {context}"},
                        {"role": "assistant", "content": self.generate_risk_analysis(analysis)}
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": "You are STEVE, an expert P&C insurance analyst."},
                        {"role": "user", "content": f"What actions should management take based on: {context}"},
                        {"role": "assistant", "content": self.generate_action_recommendations(analysis)}
                    ]
                }
            ]
            
            training_data.extend(qa_pairs)
        
        return training_data

    def build_context_string(self, analysis: Dict) -> str:
        """Build context string from analysis for training"""
        context_parts = []
        
        # Basic info
        context_parts.append(f"Data Type: {analysis.get('data_type', 'unknown')}")
        context_parts.append(f"Records: {analysis.get('rows', 0):,}")
        
        # Financial ratios
        if 'ratios' in analysis:
            ratios = analysis['ratios']
            if 'loss_ratio' in ratios:
                context_parts.append(f"Loss Ratio: {ratios['loss_ratio']:.1f}%")
            if 'reserve_to_paid_ratio' in ratios:
                context_parts.append(f"Reserve to Paid Ratio: {ratios['reserve_to_paid_ratio']:.1f}%")
        
        # Monetary data
        if 'monetary_columns' in analysis:
            for col in analysis['monetary_columns'][:3]:
                context_parts.append(f"{col['column']}: Total ${col['total']:,.0f}, Average ${col['mean']:,.0f}")
        
        # Trends
        if 'trends' in analysis:
            for col, trend in analysis['trends'].items():
                if isinstance(trend, dict):
                    context_parts.append(f"{col}: {trend['strength']} {trend['direction']} trend")
        
        # Anomalies
        if 'anomalies' in analysis:
            for anomaly in analysis['anomalies']:
                context_parts.append(f"Anomalies in {anomaly['column']}: {anomaly['outlier_percentage']:.1f}% outliers")
        
        return "; ".join(context_parts)

    def generate_expert_analysis(self, analysis: Dict) -> str:
        """Generate expert-level analysis for training data"""
        insights = []
        
        # Loss ratio analysis
        if 'ratios' in analysis and 'loss_ratio' in analysis['ratios']:
            lr = analysis['ratios']['loss_ratio']
            data_type = analysis.get('data_type', 'unknown')
            
            if 'auto' in data_type.lower():
                benchmarks = self.insurance_knowledge['loss_ratio_benchmarks']['auto']
            else:
                benchmarks = self.insurance_knowledge['loss_ratio_benchmarks']['property']
            
            if lr > benchmarks['poor']:
                insights.append(f"CRITICAL: Loss ratio of {lr:.1f}% significantly exceeds acceptable thresholds. Immediate pricing and underwriting review required.")
            elif lr > benchmarks['acceptable']:
                insights.append(f"WARNING: Loss ratio of {lr:.1f}% indicates marginal profitability. Monitor closely and consider corrective actions.")
            else:
                insights.append(f"POSITIVE: Loss ratio of {lr:.1f}% indicates healthy profitability within acceptable range.")
        
        # Trend analysis
        if 'trends' in analysis:
            for col, trend_data in analysis['trends'].items():
                if isinstance(trend_data, dict) and trend_data['strength'] == 'strong':
                    if trend_data['direction'] == 'increasing' and 'loss' in col.lower():
                        insights.append(f"CONCERN: Strong increasing trend in {col} suggests deteriorating loss experience requiring investigation.")
                    elif trend_data['direction'] == 'increasing' and 'premium' in col.lower():
                        insights.append(f"POSITIVE: Strong increasing trend in {col} indicates business growth and rate adequacy improvements.")
        
        # Data quality insights
        if 'data_quality' in analysis:
            missing_count = len(analysis['data_quality'].get('missing_data', {}))
            if missing_count > 3:
                insights.append(f"DATA QUALITY: {missing_count} columns have significant missing data, which may compromise analysis accuracy and regulatory reporting.")
        
        return " ".join(insights) if insights else "Analysis shows standard insurance metrics within expected ranges."

    def generate_risk_analysis(self, analysis: Dict) -> str:
        """Generate risk-focused analysis"""
        risks = []
        
        # Financial risks
        if 'ratios' in analysis:
            ratios = analysis['ratios']
            if ratios.get('loss_ratio', 0) > 100:
                risks.append("UNDERWRITING RISK: Unprofitable loss ratio threatens financial stability")
            
            if ratios.get('reserve_to_paid_ratio', 0) < 10:
                risks.append("RESERVE RISK: Low reserve levels may indicate under-reserving")
        
        # Volatility risks
        if 'anomalies' in analysis:
            high_anomalies = [a for a in analysis['anomalies'] if a['severity'] == 'high']
            if high_anomalies:
                risks.append(f"VOLATILITY RISK: High number of outliers in {len(high_anomalies)} columns suggests unstable loss patterns")
        
        # Operational risks
        if 'data_quality' in analysis:
            if analysis['data_quality'].get('duplicates', 0) > analysis.get('rows', 0) * 0.05:
                risks.append("OPERATIONAL RISK: Significant duplicate records indicate data management issues")
        
        return " ".join(risks) if risks else "Risk profile appears within acceptable parameters based on available data."

    def generate_action_recommendations(self, analysis: Dict) -> str:
        """Generate actionable management recommendations"""
        actions = []
        
        # Profitability actions
        if 'ratios' in analysis and analysis['ratios'].get('loss_ratio', 0) > 85:
            actions.append("PRICING: Implement rate increases focusing on worst-performing segments")
            actions.append("UNDERWRITING: Tighten underwriting guidelines and enhance risk selection")
        
        # Growth actions
        if 'trends' in analysis:
            declining_trends = [col for col, trend in analysis['trends'].items() 
                              if isinstance(trend, dict) and trend['direction'] == 'decreasing' 
                              and 'premium' in col.lower()]
            if declining_trends:
                actions.append("BUSINESS DEVELOPMENT: Address declining premium trends through market expansion or product enhancement")
        
        # Operational actions
        if 'data_quality' in analysis and analysis['data_quality'].get('missing_data'):
            actions.append("DATA MANAGEMENT: Improve data collection processes to enhance analysis accuracy")
        
        return " ".join(actions) if actions else "Continue current management practices with regular monitoring of key metrics."

    def fine_tune_openai_model(self, training_data: List[Dict], api_key: str):
        """Fine-tune OpenAI model with insurance data"""
        if not OPENAI_AVAILABLE:
            return "OpenAI library not available"
        
        try:
            openai.api_key = api_key
            
            # Create training file
            training_file = openai.files.create(
                file=open("steve_training_data.jsonl", "rb"),
                purpose="fine-tune"
            )
            
            # Start fine-tuning job
            fine_tune_job = openai.fine_tuning.jobs.create(
                training_file=training_file.id,
                model="gpt-3.5-turbo"
            )
            
            return f"Fine-tuning job started: {fine_tune_job.id}"
            
        except Exception as e:
            return f"Fine-tuning failed: {str(e)}"

    def load_custom_model(self, model_path: str):
        """Load a custom trained model"""
        try:
            if TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                return True
            return False
        except Exception as e:
            st.error(f"Failed to load custom model: {str(e)}")
            return False

    def generate_ai_response(self, question: str, context: str, model_type: str = "openai"):
        """Generate AI response using trained model"""
        
        system_prompt = f"""You are STEVE, an expert Property & Casualty insurance analyst with deep domain knowledge. 
        
        Your expertise includes:
        - Loss ratio analysis and benchmarking
        - Reserve adequacy assessment  
        - Claims trend analysis
        - Risk assessment and management
        - Regulatory compliance
        - Financial analysis for P&C insurers
        
        Current analysis context: {context}
        
        Provide detailed, actionable insights with specific recommendations."""
        
        if model_type == "openai" and OPENAI_AVAILABLE:
            try:
                response = openai.chat.completions.create(
                    model="gpt-4",  # Or your fine-tuned model ID
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=800,
                    temperature=0.3  # Lower temperature for more consistent analysis
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"AI response failed: {str(e)}"
        
        elif model_type == "custom" and hasattr(self, 'model'):
            try:
                inputs = self.tokenizer.encode(f"{system_prompt}\n\nUser: {question}\n\nSTEVE:", return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs, 
                        max_length=inputs.shape[1] + 200,
                        num_return_sequences=1,
                        temperature=0.3,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response.split("STEVE:")[-1].strip()
            except Exception as e:
                return f"Custom model response failed: {str(e)}"
        
        else:
            return "No AI model available. Please configure an API key or load a custom model."

# Training Data Management Functions
def save_training_data(training_data: List[Dict], filename: str = "steve_training_data.jsonl"):
    """Save training data in JSONL format for fine-tuning"""
    with open(filename, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    return filename

def create_insurance_benchmark_dataset():
    """Create a comprehensive benchmark dataset for insurance analysis"""
    benchmark_data = [
        {
            "scenario": "auto_insurance_high_loss_ratio",
            "data": {"loss_ratio": 105, "claim_count": 1500, "avg_claim": 8500, "data_type": "auto"},
            "expert_analysis": "Critical situation: Auto insurance loss ratio of 105% indicates significant unprofitability. With 1,500 claims averaging $8,500, this suggests either inadequate pricing or adverse selection. Immediate actions required: 1) Implement rate increases of 15-20%, 2) Review underwriting guidelines, 3) Analyze claim patterns for fraud, 4) Consider market withdrawal from unprofitable segments."
        },
        {
            "scenario": "property_insurance_storm_losses",
            "data": {"loss_ratio": 140, "catastrophe_losses": 2500000, "claim_count": 300, "data_type": "property"},
            "expert_analysis": "Catastrophe event impact: Property loss ratio of 140% driven by $2.5M in storm losses across 300 claims. This is expected for catastrophe years. Recommendations: 1) Verify reinsurance recovery, 2) Assess reserve adequacy for remaining development, 3) Review catastrophe modeling and pricing, 4) Consider geographic risk concentration limits."
        }
    ]
    return benchmark_data

# Example usage in Streamlit app
def integrate_ai_training_ui():
    """UI components for AI training integration"""
    
    st.sidebar.header("🤖 AI Training Options")
    
    training_option = st.sidebar.selectbox(
        "Choose Training Approach:",
        ["No AI Training", "Fine-tune OpenAI", "Load Custom Model", "Create Training Data"]
    )
    
    if training_option == "Fine-tune OpenAI":
        api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
        if st.sidebar.button("Start Fine-tuning"):
            if api_key and st.session_state.analyzed_data:
                trainer = SteveAITrainer()
                training_data = trainer.create_insurance_fine_tuning_dataset(st.session_state.analyzed_data)
                
                # Save training data
                filename = save_training_data(training_data)
                st.success(f"Training data created: {len(training_data)} examples")
                
                # Start fine-tuning
                result = trainer.fine_tune_openai_model(training_data, api_key)
                st.info(result)
    
    elif training_option == "Create Training Data":
        if st.sidebar.button("Generate Training Dataset"):
            if st.session_state.analyzed_data:
                trainer = SteveAITrainer()
                training_data = trainer.create_insurance_fine_tuning_dataset(st.session_state.analyzed_data)
                
                filename = save_training_data(training_data)
                st.success(f"Created {filename} with {len(training_data)} training examples")
                
                # Show sample
                st.json(training_data[0])
    
    return training_option

# This shows how to integrate the trained AI into the main STEVE application
print("🤖 STEVE AI Training Framework Ready!")
print("Options: Fine-tuning, Custom Models, Training Data Generation")
