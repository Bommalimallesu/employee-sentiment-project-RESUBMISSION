
import pandas as pd, numpy as np
from transformers import pipeline

_sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def pick_column(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for c in df.columns:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None

def detect_columns(df):
    emp_col = pick_column(df, ["employee_id","employee","user","sender","email","from","emp_id","name","author"])
    text_col = pick_column(df, ["message","text","content","email_body","body","comment","remark"])
    time_col = pick_column(df, ["timestamp","datetime","date","time","created_at","sent_at","ts"])
    return emp_col, text_col, time_col

def normalize_dataframe(df):
    emp_col, text_col, time_col = detect_columns(df)
    if text_col is None:
        raise ValueError("No message column found")
    if emp_col is None:
        df["_employee"] = "EMP_" + (df.index+1).astype(str)
        emp_col = "_employee"
    if time_col is None:
        base = pd.Timestamp("2024-01-01")
        df["_timestamp"] = [base + pd.Timedelta(days=i%90) for i in range(len(df))]
        time_col = "_timestamp"
    out = df.rename(columns={emp_col:"employee", text_col:"message", time_col:"timestamp"})[["employee","message","timestamp"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["message","timestamp"])
    return out

def sentiment_score_llm(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    res = _sentiment_model(text[:512])[0]
    return res["score"] if res["label"]=="POSITIVE" else -res["score"]

def label_from_score(score: float) -> str:
    if score > 0.15: return "Positive"
    if score < -0.15: return "Negative"
    return "Neutral"

def add_sentiment(df):
    df = df.copy()
    df["sentiment_score"] = df["message"].apply(sentiment_score_llm)
    df["sentiment_label"] = df["sentiment_score"].apply(label_from_score)
    return df

def monthly_scores(df):
    df = df.copy()
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["msg_score"] = df["sentiment_label"].map({"Positive":1,"Negative":-1,"Neutral":0}).fillna(0)
    agg = df.groupby(["employee","month"]).agg(
        monthly_score=("msg_score","sum"),
        message_count=("message","count"),
        avg_len=("message", lambda s: float(np.mean([len(x.split()) for x in s]))),
        neg_count=("sentiment_label", lambda s: int((s=="Negative").sum())),
        pos_count=("sentiment_label", lambda s: int((s=="Positive").sum())),
        neu_count=("sentiment_label", lambda s: int((s=="Neutral").sum()))
    ).reset_index()
    agg["neg_rate"] = (agg["neg_count"] / agg["message_count"]).replace([np.inf,np.nan],0.0)
    return agg

def top3_by_month(agg):
    out={}
    for m,g in agg.groupby("month"):
        pos = g.sort_values(["monthly_score","employee"], ascending=[False,True]).head(3)
        neg = g.sort_values(["monthly_score","employee"], ascending=[True,True]).head(3)
        out[m]=(pos,neg)
    return out

def flight_risk(df):
    df = df.copy().sort_values(["employee","timestamp"])
    df["is_negative"] = (df["sentiment_label"]=="Negative").astype(int)
    risks=set()
    for emp,g in df.groupby("employee"):
        dates = g.loc[g["is_negative"]==1,"timestamp"].sort_values().to_list()
        if len(dates)<4: continue
        i=0
        for j in range(len(dates)):
            while dates[j]-dates[i] > pd.Timedelta(days=30):
                i+=1
            if j-i+1>=4:
                risks.add(emp); break
    return sorted(list(risks))

def gen_features_for_regression(agg):
    import numpy as np
    features=agg[["message_count","avg_len","neg_rate","pos_count","neg_count"]].copy()
    y=agg["monthly_score"].values
    return features,y


_POS = set(["good","great","happy","love","excellent","awesome","thanks","thank you","well done","improved","promotion","success"])
_NEG = set(["bad","poor","sad","angry","frustrated","blocked","issue","bug","unhappy","hate","delay","failed","failure","overworked"])

def baseline_sentiment_rule(text: str) -> float:
    import re
    if not isinstance(text, str) or not text.strip():
        return 0.0
    t = text.lower()
    pos_hits = sum(1 for w in _POS if re.search(r'\b'+re.escape(w)+r'\b', t))
    neg_hits = sum(1 for w in _NEG if re.search(r'\b'+re.escape(w)+r'\b', t))
    score = 0.0
    if pos_hits or neg_hits:
        score = (pos_hits - neg_hits) / max(1, (pos_hits + neg_hits))
    return max(-1.0, min(1.0, float(score)))
