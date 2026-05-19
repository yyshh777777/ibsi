import streamlit as st
import pandas as pd
import chromadb
from openai import OpenAI
import os
from chromadb.utils import embedding_functions

# ==========================================
# 1. 페이지 설정 및 디자인 (기존 UI 유지 및 개선)
# ==========================================
st.set_page_config(page_title="입시 컨설팅 AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    /* 전체 폰트 및 배경 */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 채팅창 스타일 */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e3f2fd; /* 사용자 질문 배경색 (연한 파랑) */
        border: 1px solid #bbdefb;
    }
    
    /* 헤더 스타일 */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1565c0;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 대입 합격예측 AI 컨설턴트</div>', unsafe_allow_html=True)

# ==========================================
# 2. API 설정 및 안전한 키 로드
# ==========================================
try:
    # 로컬환경(.streamlit/secrets.toml) 및 Streamlit Cloud Secrets에서 키 로드
    api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("⚠️ Secrets 설정이 되어있지 않습니다. 사이드바 설정을 확인하거나 .streamlit/secrets.toml 파일을 확인하세요.")
    st.stop()

# ==========================================
# 3. CSV 데이터 자동 로드 및 ChromaDB 빌드
# ==========================================
CSV_FILE_PATH = "ibsi.csv"

@st.cache_resource
def init_vector_db(_api_key):
    """CSV 파일을 읽어서 ChromaDB에 자동으로 인덱싱하는 함수"""
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=_api_key,
        model_name="text-embedding-3-small"
    )
    
    # 임베딩 전용 고유 클라이언트 생성 (임시/인메모리 방식 혹은 가벼운 로컬 디렉토리)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="admissions_idx", embedding_function=openai_ef)
    
    # DB에 데이터가 비어있는 경우에만 CSV 데이터 삽입 (속도 최적화)
    if collection.count() == 0:
        if not os.path.exists(CSV_FILE_PATH):
            st.error(f"❌ '{CSV_FILE_PATH}' 파일이 루트 디렉토리에 존재하지 않습니다. 파일을 배치해 주세요.")
            st.stop()
            
        df = pd.read_csv(CSV_FILE_PATH)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, row in df.iterrows():
            # AI가 읽기 좋은 형태의 텍스트 문서 생성
            doc_text = (
                f"대학: {row['학교명']}, 전형: {row['전형']}, 학과(모집단위): {row['모집단위']}, "
                f"50% 커트라인: {row['50% cut']}등급, 70% 커트라인: {row['70% cut']}등급, "
                f"반영교과: {row['평가에 반영된 교과목']}"
            )
            documents.append(doc_text)
            
            # 필터링에 사용할 메타데이터 구성 (공백 제거하여 매칭 확률 상승)
            metadatas.append({
                "학교명": str(row['학교명']).strip(),
                "전형": str(row['전형']).strip(),
                "모집단위": str(row['모집단위']).strip()
            })
            ids.append(f"id_{i}")
            
        # 데이터 분할 삽입 (ChromaDB 안정성 확보)
        batch_size = 100
        for idx in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[idx:idx+batch_size],
                metadatas=metadatas[idx:idx+batch_size],
                ids=ids[idx:idx+batch_size]
            )
            
    return collection

# 데이터베이스 초기화 실행
collection = init_vector_db(api_key)

# ==========================================
# 4. 사이드바 UI 및 필터 옵션 추출
# ==========================================
@st.cache_data
def get_filter_options():
    """사이드바 필터링을 위해 CSV에서 고유값 추출"""
    if os.path.exists(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH)
        schools = sorted(df['학교명'].dropna().unique().tolist())
        types = sorted(df['전형'].dropna().unique().tolist())
        return schools, types
    return [], []

school_list, type_list = get_filter_options()

with st.sidebar:
    st.header("📋 학생 프로필 설정")
    
    with st.expander("🏫 목표 대학 및 전형", expanded=True):
        target_school = st.selectbox("희망 대학", ["전체"] + school_list)
        selected_type = st.selectbox("희망 전형", ["전체"] + type_list)

    st.markdown("---")
    
    with st.container():
        st.subheader("📊 나의 성적")
        col1, col2 = st.columns(2)
        with col1:
            my_grade = st.number_input("내신 등급", 1.00, 9.00, 3.00, 0.1, format="%.2f")
        with col2:
            record_level = st.select_slider(
                "생기부 수준", 
                options=["하", "중", "상", "최상"], 
                value="중"
            )
        
        st.info(f"현재 설정: **{my_grade}등급** / 생기부 **{record_level}**")
        st.caption("💡 숫자가 작을수록(1.0) 우수한 성적입니다.")

# ==========================================
# 5. 메인 채팅 로직 (대화 기억 기능 탑재)
# ==========================================
# 세션 상태에 대화 기록 보존 장치 마련
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 👋\n업로드된 공식 입시 데이터를 바탕으로 합격 가능성을 정확하게 예측해 드립니다.\n궁금한 대학이나 학과를 말씀해 주세요! (예: 부산대 국어교육과 가능할까요?)"}
    ]

# 대화 기록 렌더링
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 유저 입력창 처리
if prompt := st.chat_input("질문을 입력해 주세요..."):
    # 1. 유저 발언 기록 및 화면 출력
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 데이터베이스에서 매칭되는 입시 정보 조회 중..."):
            
            # 동적 metadata 필터링 조건 설정
            where_conditions = []
            if target_school != "전체":
                where_conditions.append({"학교명": target_school})
            if selected_type != "전체":
                where_conditions.append({"전형": selected_type})

            final_where = None
            if len(where_conditions) == 1:
                final_where = where_conditions[0]
            elif len(where_conditions) > 1:
                final_where = {"$and": where_conditions}

            try:
                # 관련 데이터 5건 검색
                results = collection.query(query_texts=[prompt], n_results=5, where=final_where)
                docs = results['documents'][0] if results['documents'] else []
                
                context = ""
                if docs:
                    for i, doc in enumerate(docs):
                        context += f"[입시데이터 {i+1}] {doc}\n"
                else:
                    context = "조건에 부합하는 정형화된 입시 데이터가 전혀 존재하지 않습니다."

                # 프롬프트 엔지니어링: 냉철한 분석가 페르소나 및 추론 금지 조항 강화
                system_prompt = f"""
                당신은 대한민국 최고 권위의 냉철하고 객관적인 대입 합격 예측 분석가입니다.
                제공된 명확한 실측 데이터에만 근거하여 답변해야 하며, 없는 데이터를 가공하거나 임의로 '추론', '예측', '상상'해서 말하는 것을 절대 금지합니다.

                [학생의 현재 프로필]
                - 내신 등급: {my_grade}등급 (1.0에 가까울수록 최상위 성적, 9.0에 가까울수록 최하위 성적)
                - 학교생활기록부 상태: {record_level}

                [수학적 평가 절대 원칙]
                - 공식: (학생 등급 - 데이터상의 커트라인 등급) = 차이값
                - 차이값이 양수(+)이면: 학생의 등급 숫자가 더 큼 -> 성적이 미달함 -> 위험 또는 상향 지원 판정.
                - 차이값이 음수(-)이면: 학생의 등급 숫자가 더 작음 -> 성적이 여유 있음 -> 안정 또는 적정 지원 판정.
                
                [★ 초정밀 RAG 지침 (필수)]
                1. 제공된 [참고 데이터]에 사용자가 질문한 대학교나 학과의 컷(50% cut, 70% cut) 정보가 없다면, 절대로 "합격 가능성이 높다/낮다" 혹은 "이 점수대면 합격선일 것이다" 같은 가상의 추론 답변을 하지 마십시오.
                2. 데이터가 없다면 정확하게 "제공된 데이터베이스에 해당 학과/대학에 대한 공식 컷 데이터가 부재하여 안내가 불가능합니다."라고만 명확히 선을 그으십시오.

                [제공된 실제 입시 데이터]
                {context}

                [답변 필수 양식]
                1. **판정 결과:** (안정 / 적정 / 소신 / 상향 / 위험 / 데이터 부족으로 판정 불가 중 택1)
                2. **상세 분석:** (데이터에 기재된 50% cut, 70% cut 수치와 학생의 등급을 수학적으로 명확히 비교 계산하여 서술)
                3. **전략적 조언:** (생기부 수준 및 해당 전형의 특징에 맞춘 현실적인 행동 지침 제공)
                """

                # 대화 내역 기억 장치 연동 (최근 대화 기록 포함하여 전송)
                msgs = [{"role": "system", "content": system_prompt}]
                
                # 메모리 과부하 및 토큰 절약을 위해 시스템 프롬프트 외 최근 6개의 대화 쌍만 컨텍스트로 전달
                msgs.extend(st.session_state.messages[-6:])

                # OpenAI API 호출
                client = OpenAI(api_key=api_key)
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=msgs,
                    temperature=0.0  # 추론을 억제하고 정량적 정확도를 극대화하기 위해 0.0 설정
                )
                answer = res.choices[0].message.content

            except Exception as e:
                answer = f"⚠️ 입시 데이터를 분석하는 중 예기치 못한 에러가 발생했습니다: {e}"

            # 화면 출력 및 메모리에 저장
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})