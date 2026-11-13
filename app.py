import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="EcoTrack AI", page_icon="🚗", layout="wide")
st.title("🚗 EcoTrack AI")
st.caption("자동차 배출가스 데이터 분석 및 AI 이상 패턴 탐지 프로토타입")

st.info("이 앱은 학교 프로젝트용 분석 시제품입니다. AI 이상 패턴은 법적 위반이나 차량 고장을 뜻하지 않습니다.")

with st.expander("분석 시 주의사항"):
    st.markdown("""
    - 무부하검사, 부하검사(ASM-Idling·KD-147), 제작차 인증시험, 도로 원격측정(RSD)은 서로 다른 측정 방식입니다.
    - 측정 방식·차량 조건·단위가 일치하지 않는 기준을 비교하면 안 됩니다.
    - CO₂는 온실가스 지표이며 CO·HC·NOx와 같은 의미의 오염물질 기준으로 취급하면 안 됩니다.
    - 이 앱은 이상 패턴을 선별할 뿐, 법적 처분을 자동으로 결정하지 않습니다.
    """)

def load_csv(f):
    b = f.getvalue()
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc)
        except UnicodeDecodeError:
            pass
    raise ValueError("CSV 인코딩을 읽지 못했습니다. UTF-8 또는 CP949로 저장해 주세요.")

def numeric_cols(df):
    return [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]

st.sidebar.header("데이터 업로드")
measurement_file = st.sidebar.file_uploader("① 배출가스 측정 데이터 CSV", type="csv")
standard_file = st.sidebar.file_uploader("② 배출허용기준 CSV (선택)", type="csv")

if measurement_file is None:
    st.markdown("""
    ### 사용 순서
    1. 측정 데이터 CSV 업로드
    2. 실제 CSV의 열 이름에 맞춰 오염물질 열 선택
    3. 기초 통계와 분포 확인
    4. AI 이상 패턴 탐지 실행
    5. 검사 방식이 일치하는 경우에만 기준값 비교
    """)
    st.stop()

try:
    df = load_csv(measurement_file)
except Exception as e:
    st.error(str(e))
    st.stop()

if df.empty:
    st.error("CSV에 데이터가 없습니다.")
    st.stop()

st.subheader("1. 측정 데이터")
a, b, c = st.columns(3)
a.metric("관측 건수", f"{len(df):,}")
b.metric("열 개수", f"{len(df.columns):,}")
c.metric("중복 행", f"{df.duplicated().sum():,}")
st.dataframe(df.head(30), use_container_width=True)

nums = numeric_cols(df)
if not nums:
    st.error("숫자형 측정값 열이 없습니다.")
    st.stop()

st.subheader("2. 오염물질 열 지정")
aliases = {
    "CO": ["co", "일산화탄소"],
    "HC": ["hc", "탄화수소"],
    "CO₂": ["co2", "이산화탄소"],
    "NOx": ["nox", "질소산화물"],
    "O₂": ["o2", "산소"],
}
mapped = {}
cols = st.columns(2)
for i, (name, keys) in enumerate(aliases.items()):
    guess = next((col for col in df.columns
                  if any(k in str(col).lower().replace(" ", "") for k in keys)), None)
    opts = ["선택 안 함"] + list(df.columns.astype(str))
    idx = opts.index(str(guess)) if guess is not None and str(guess) in opts else 0
    choice = cols[i % 2].selectbox(f"{name} 측정값 열", opts, index=idx, key=f"map_{name}")
    if choice != "선택 안 함":
        mapped[name] = choice

if mapped:
    pollutant = st.selectbox("분포 그래프로 볼 항목", list(mapped.keys()))
    values = pd.to_numeric(df[mapped[pollutant]], errors="coerce").dropna()
    if len(values):
        st.plotly_chart(px.histogram(x=values, nbins=30,
            labels={"x": f"{pollutant} 측정값", "y": "관측 건수"},
            title=f"{pollutant} 측정값 분포"), use_container_width=True)
        st.dataframe(values.describe().to_frame("통계값"), use_container_width=True)

st.subheader("3. AI 이상 패턴 탐지")
st.write("Isolation Forest 비지도 학습으로 여러 측정 변수의 조합이 데이터의 일반적인 분포와 다른 관측값을 탐색합니다.")
features = st.multiselect("AI 분석 변수", nums,
                          default=[v for v in mapped.values() if v in nums])
contamination = st.slider("이상치 비율 설정값", 0.01, 0.20, 0.05, 0.01,
                          help="실제 이상 차량 비율이 아니라 알고리즘의 탐지 설정값입니다.")
if st.button("AI 분석 실행", type="primary", disabled=not features):
    if len(df) < 20:
        st.error("관측값이 20건 미만입니다. 더 많은 데이터 확보를 권장합니다.")
    else:
        X = df[features].apply(pd.to_numeric, errors="coerce")
        X = X[[col for col in X if X[col].notna().any()]]
        if X.shape[1] == 0:
            st.error("선택한 열에 유효한 숫자 데이터가 없습니다.")
        else:
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                IsolationForest(n_estimators=200, contamination=contamination, random_state=42))
            labels = pipe.fit_predict(X)
            scores = pipe.decision_function(X)
            result = df.copy()
            result["AI_분석결과"] = np.where(labels == -1, "추가 확인이 필요한 이상 패턴", "일반 패턴")
            result["AI_이상점수"] = scores
            st.session_state["ai_result"] = result

if "ai_result" in st.session_state:
    result = st.session_state["ai_result"]
    abnormal = int((result["AI_분석결과"] == "추가 확인이 필요한 이상 패턴").sum())
    x, y = st.columns(2)
    x.metric("일반 패턴", f"{len(result)-abnormal:,}건")
    y.metric("추가 확인 필요", f"{abnormal:,}건")
    st.dataframe(result, use_container_width=True)
    st.download_button("AI 결과 CSV 다운로드",
        result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        "ecotrack_ai_result.csv", "text/csv")
    st.warning("이상 패턴은 법적 기준 초과나 차량 결함의 증거가 아닙니다. 추가 확인 후보를 찾는 참고 결과입니다.")

st.subheader("4. 배출허용기준 확인 및 비교")
if standard_file is not None:
    try:
        standards = load_csv(standard_file)
        st.write("업로드한 기준표 미리보기")
        st.dataframe(standards.head(50), use_container_width=True)
    except Exception as e:
        st.error(f"기준표 읽기 오류: {e}")

if mapped:
    pollutant = st.selectbox("기준과 비교할 측정 항목", list(mapped.keys()), key="compare_pollutant")
    threshold = st.number_input("공식 자료에서 확인한 기준값 입력", min_value=0.0, value=0.0, format="%.6f")
    conditions_match = st.checkbox("검사 방법·시험 조건·차량 조건·측정 단위가 모두 일치함을 확인했습니다.")
    if st.button("기준값 비교"):
        if not conditions_match:
            st.error("적용 조건이 일치하는지 먼저 확인해야 합니다.")
        elif threshold <= 0:
            st.error("0보다 큰 기준값을 입력해 주세요.")
        else:
            out = df.copy()
            measured = pd.to_numeric(df[mapped[pollutant]], errors="coerce")
            out["측정값"] = measured
            out["입력한_기준값"] = threshold
            out["비교결과"] = np.where(measured.isna(), "측정값 없음",
                np.where(measured > threshold, "입력 기준값 초과", "입력 기준값 이내"))
            st.dataframe(out, use_container_width=True)
            st.download_button("기준 비교 결과 CSV 다운로드",
                out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                "ecotrack_threshold_result.csv", "text/csv")
            st.caption("사용자가 입력한 기준값과의 참고 비교입니다. 법적 판정을 자동으로 내리지 않습니다.")

st.divider()
st.caption("EcoTrack AI | 학교 프로젝트용 시제품")
