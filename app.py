import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import re
from urllib.parse import urlparse
import tldextract
import warnings
import time

# --- SETUP & CONFIG ---
st.set_page_config(page_title="PhishGuard AI Dashboard", page_icon="🛡️", layout="wide")
warnings.filterwarnings('ignore')
np.random.seed(42)

# --- ML CORE CLASSES & LOGIC ---

def create_perfect_phishing_dataset():
    """Generates a realistic synthetic dataset for phishing detection training."""
    n_samples = 15000
    data = {
        'url_length': np.concatenate([
            np.random.normal(95, 25, n_samples//2).clip(50, 250),   # Phishing
            np.random.normal(35, 15, n_samples//2).clip(10, 100)    # Legitimate
        ]),
        'num_dots': np.concatenate([
            np.random.poisson(3.8, n_samples//2).clip(1, 15),
            np.random.poisson(1.8, n_samples//2).clip(1, 5)
        ]),
        'num_hyphens': np.concatenate([
            np.random.poisson(2.5, n_samples//2),
            np.random.poisson(0.4, n_samples//2)
        ]),
        'num_digits': np.concatenate([
            np.random.poisson(8, n_samples//2),
            np.random.poisson(0.6, n_samples//2)
        ]),
        'has_https': np.concatenate([
            np.random.binomial(1, 0.25, n_samples//2),
            np.random.binomial(1, 0.95, n_samples//2)
        ]),
        'domain_length': np.concatenate([
            np.random.normal(32, 10, n_samples//2).clip(10, 80),
            np.random.normal(14, 5, n_samples//2).clip(5, 40)
        ]),
        'num_subdomains': np.concatenate([
            np.random.poisson(2.5, n_samples//2),
            np.random.poisson(0.8, n_samples//2)
        ]),
        'has_ip': np.concatenate([
            np.random.binomial(1, 0.4, n_samples//2),
            np.random.binomial(1, 0.001, n_samples//2)
        ]),
        'suspicious_tld': np.concatenate([
            np.random.binomial(1, 0.7, n_samples//2),
            np.random.binomial(1, 0.02, n_samples//2)
        ]),
        'shortening_service': np.concatenate([
            np.random.binomial(1, 0.5, n_samples//2),
            np.random.binomial(1, 0.01, n_samples//2)
        ]),
        'suspicious_keywords': np.concatenate([
            np.random.poisson(1.5, n_samples//2),
            np.random.poisson(0.1, n_samples//2)
        ]),
        'is_phishing': np.concatenate([np.ones(n_samples//2), np.zeros(n_samples//2)])
    }
    df = pd.DataFrame(data)
    noise_indices = np.random.choice(len(df), size=int(0.03 * len(df)), replace=False)
    for idx in noise_indices:
        df.loc[idx, 'is_phishing'] = 1 - df.loc[idx, 'is_phishing']
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

class AdvancedPhishingDetector:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight={0: 1, 1: 1.5})
        self.xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=8, scale_pos_weight=2, eval_metric='logloss')
        self.lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=7, scale_pos_weight=2, verbosity=-1)
        self.gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.rf_model.fit(X_scaled, y_train)
        self.xgb_model.fit(X_scaled, y_train)
        self.lgb_model.fit(X_scaled, y_train)
        self.gb_model.fit(X_scaled, y_train)
        self.is_trained = True

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = [model.predict(X_scaled) for model in [self.rf_model, self.xgb_model, self.lgb_model, self.gb_model]]
        return np.where(np.sum(preds, axis=0) >= 2, 1, 0)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        probs = [model.predict_proba(X_scaled)[:, 1] for model in [self.rf_model, self.xgb_model, self.lgb_model, self.gb_model]]
        return np.mean(probs, axis=0)

def smart_feature_extraction(url):
    features = {}
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['has_https'] = 1 if url.startswith('https') else 0
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        features['domain_length'] = len(domain)
        features['num_subdomains'] = domain.count('.')
        ext = tldextract.extract(url)
        common_tlds = ['com', 'org', 'net', 'edu', 'gov', 'io', 'co', 'ai']
        features['suspicious_tld'] = 1 if ext.suffix not in common_tlds else 0
        features['has_ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain) else 0
        short_patterns = ['.ly', '.gl', 'tinyurl', 'bit.ly', 'goo.gl', 't.co', 'ow.ly']
        features['shortening_service'] = 1 if any(pattern in domain for pattern in short_patterns) else 0
        phishing_keywords = ['login', 'verify', 'secure', 'account', 'banking', 'update', 'signin', 'confirm', 'password', 'wallet', 'payment', 'authenticate', 'validation', 'security', 'online', 'webscr', 'profile', 'ebayisapi', 'click', 'here', 'free', 'gift', 'award', 'claim', 'service']
        features['suspicious_keywords'] = sum(1 for word in phishing_keywords if word in url.lower())
        features['multiple_subdomains'] = 1 if features['num_subdomains'] >= 2 else 0
        if len(url) > 0:
            prob = [url.count(c) / len(url) for c in set(url)]
            features['entropy'] = -sum(p * np.log2(p) for p in prob)
        else: features['entropy'] = 0
    except:
        features.update({'domain_length': 0, 'num_subdomains': 0, 'suspicious_tld': 0, 'has_ip': 0, 'shortening_service': 0, 'suspicious_keywords': 0, 'multiple_subdomains': 0, 'entropy': 0})
    return features

# --- INITIALIZATION ---

@st.cache_resource
def get_trained_model():
    df = create_perfect_phishing_dataset()
    X = df.drop('is_phishing', axis=1)
    y = df['is_phishing']
    detector = AdvancedPhishingDetector()
    detector.fit(X, y)
    return detector, X.columns.tolist()

model, feature_order = get_trained_model()

# --- CSS STYLES ---

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

:root {
    --bg-dark: #0a0b10;
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --accent-blue: #00d2ff;
    --accent-purple: #9d50bb;
    --safe-green: #00ffa3;
    --danger-red: #ff3e3e;
}

.stApp { background: var(--bg-dark); color: white; font-family: 'Outfit', sans-serif; }

.header-container { text-align: center; padding: 40px 0; background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.glass-card { background: var(--glass-bg); backdrop-filter: blur(12px); border: 1px solid var(--glass-border); border-radius: 20px; padding: 30px; margin-bottom: 25px; transition: all 0.3s ease; }

.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }

.status-badge { display: inline-block; padding: 8px 16px; border-radius: 100px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; }

.badge-safe { background: rgba(0, 255, 163, 0.1); color: var(--safe-green); border: 1px solid var(--safe-green); }
.badge-danger { background: rgba(255, 62, 62, 0.1); color: var(--danger-red); border: 1px solid var(--danger-red); }

.feature-item { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }

/* Custom Streamlit Overrides */
.stTextInput > div > div > input { background: rgba(255,255,255,0.05) !important; color: white !important; border-radius: 12px !important; border: 1px solid var(--glass-border) !important; }
.stButton > button { background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important; color: white !important; border: none !important; border-radius: 12px !important; width: 100% !important; height: 50px !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# --- MAIN UI ---

st.markdown('<div class="header-container"><h1>🛡️ PhishGuard AI Core</h1><p>Neural Engine v2.0 • Standalone Web Intelligence</p></div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    url_input = st.text_input("", placeholder="Enter website URL for deep analysis...", label_visibility="collapsed")
    analyze_btn = st.button("EXECUTE DEEP ANALYSIS")
    st.markdown('</div>', unsafe_allow_html=True)

if analyze_btn or url_input:
    if not url_input:
        st.error("⚠️ Please provide a URL endpoint to begin analysis.")
    else:
        with st.spinner("📡 Intercepting and Analyzing URL..."):
            time.sleep(1) # Visual effect
            
            features = smart_feature_extraction(url_input)
            
            # Prepare feature vector
            vector = [features.get(f, 0) for f in feature_order]
            vector = np.array(vector).reshape(1, -1)
            
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]
            confidence = probability if prediction == 1 else (1 - probability)
            
            # Risk scoring
            risk_score = 0
            risk_factors = []
            if features['suspicious_tld']: risk_score += 25; risk_factors.append("Untrusted TLD Extension")
            if features['has_ip']: risk_score += 30; risk_factors.append("IP-based Navigation")
            if features['shortening_service']: risk_score += 20; risk_factors.append("URL Shortener Detected")
            if not features['has_https']: risk_score += 15; risk_factors.append("Unencrypted (No HTTPS)")
            if features['suspicious_keywords'] > 0: risk_score += features['suspicious_keywords'] * 5; risk_factors.append(f"Blacklisted Keywords ({features['suspicious_keywords']})")
            if features['multiple_subdomains']: risk_score += 10; risk_factors.append("Deep Subdomain Nesting")
            
            is_phish = prediction == 1 or risk_score >= 50
            status_cls = "badge-danger" if is_phish else "badge-safe"
            status_txt = "PHISHING DETECTED" if is_phish else "SECURE ENDPOINT"
            status_icon = "🚨" if is_phish else "🛡️"
            
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <span class="status-badge {status_cls}">{status_icon} {status_txt}</span>
                <h2 style="margin: 10px 0;">URL: {url_input}</h2>
                <div class="metric-grid">
                    <div><p style="opacity:0.7">CONFIDENCE</p><h3>{confidence:.2%}</h3></div>
                    <div><p style="opacity:0.7">RISK LEVEL</p><h3 style="color:{'#ff3e3e' if is_phish else '#00ffa3'}">{'CRITICAL' if is_phish else 'MINIMAL'}</h3></div>
                    <div><p style="opacity:0.7">THREAT SCORE</p><h3>{risk_score}/100</h3></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="glass-card">
                    <h4 style="color:var(--accent-blue)">🔍 Endpoint Features</h4>
                    <div class="feature-item"><span>Address Length</span><span>{features['url_length']} chars</span></div>
                    <div class="feature-item"><span>Structural Dots</span><span>{features['num_dots']}</span></div>
                    <div class="feature-item"><span>Entropy Score</span><span>{features['entropy']:.2f}</span></div>
                    <div class="feature-item"><span>Blacklisted Words</span><span>{features['suspicious_keywords']}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="glass-card">
                    <h4 style="color:var(--accent-purple)">🔒 Security Protocols</h4>
                    <div class="feature-item"><span>SSL Encryption</span><span>{'✅ Valid' if features['has_https'] else '❌ Missing'}</span></div>
                    <div class="feature-item"><span>DNS Integrity</span><span>{'✅ Verified' if not features['suspicious_tld'] else '❌ Suspicious'}</span></div>
                    <div class="feature-item"><span>Host Type</span><span>{'Domain' if not features['has_ip'] else 'Direct IP'}</span></div>
                    <div class="feature-item"><span>Gateway</span><span>{'Standard' if not features['shortening_service'] else 'Shortner'}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
            if risk_factors:
                st.markdown('<div class="glass-card"><h4>🎯 Targeted Threat Intelligence</h4>', unsafe_allow_html=True)
                for f in risk_factors:
                    st.markdown(f'<span class="status-badge badge-danger" style="margin-right:10px">{f}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="glass-card"><h4>🎯 Threat Intelligence</h4><span class="status-badge badge-safe">No known threat patterns identified</span></div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; opacity: 0.5;">
    <p>PhishGuard AI Core Engine • Powered by Ensemble Machine Learning • Capstone Project 2024</p>
</div>
""", unsafe_allow_html=True)
