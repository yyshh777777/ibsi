import streamlit as st
import chromadb
from openai import OpenAI
import os
from chromadb.utils import embedding_functions

# ==========================================
# 1. 페이지 설정 및 디자인 (UI 개선)
# ==========================================
st.set_page_config(page_title="입시 컨설팅 AI", page_icon="🎓", layout="wide")

# CSS로 디자인 꾸미기
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
# 2. DB 및 API 설정
# ==========================================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("🔑 OpenAI API Key 입력", type="password")

if not api_key:
    st.warning("⚠️ 왼쪽 사이드바에 API 키를 입력해주세요.")
    st.stop()

@st.cache_resource
def get_collection(_api_key):
    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=_api_key,
            model_name="text-embedding-3-small"
        )
        client = chromadb.PersistentClient(path="./chroma_db")
        col = client.get_collection(name="admissions", embedding_function=openai_ef)
        return col
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        return None

collection = get_collection(api_key)
if not collection: st.stop()

# ==========================================
# 3. 데이터 로드 및 필터링
# ==========================================
@st.cache_data
def get_filter_options():
    try:
        data = collection.get(include=["metadatas"])
        school_set = set()
        type_map = {} 
        
        for meta in data['metadatas']:
            if "학교명" in meta and meta["학교명"]:
                school_set.add(meta["학교명"])
            
            if "전형" in meta and meta["전형"]:
                raw_val = meta["전형"]
                clean_name = raw_val.replace(" ", "")
                if clean_name not in type_map:
                    type_map[clean_name] = []
                if raw_val not in type_map[clean_name]:
                    type_map[clean_name].append(raw_val)

        return sorted(list(school_set)), type_map
    except:
        return [], {}

school_list, type_mapping = get_filter_options()
display_types = ["전체"] + sorted(list(type_mapping.keys()))

# ==========================================
# 4. 사이드바 UI (프로필 카드 형태)
# ==========================================
with st.sidebar:
    st.header("📋 학생 프로필 설정")
    
    with st.expander("🏫 목표 대학 및 전형", expanded=True):
        target_school = st.selectbox("희망 대학", ["전체"] + school_list)
        selected_display_type = st.selectbox("희망 전형", display_types)

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
        
        # 시각적 피드백
        st.info(f"현재 설정: **{my_grade}등급** / 생기부 **{record_level}**")
        st.caption("💡 숫자가 작을수록(1.0) 좋은 성적임을 AI가 계산합니다.")

# ==========================================
# 5. 메인 채팅 로직
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 👋\n성적과 생기부를 분석하여 합격 가능성을 예측해 드립니다.\n궁금한 학과나 대학을 물어보세요!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("질문 입력 (예: 컴퓨터공학과 가능할까요?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 입시 데이터 정밀 분석 중..."):
            
            # 필터링
            where_conditions = []
            if target_school != "전체":
                where_conditions.append({"학교명": target_school})
            
            if selected_display_type != "전체":
                real_db_values = type_mapping[selected_display_type]
                if len(real_db_values) == 1:
                    where_conditions.append({"전형": real_db_values[0]})
                else:
                    where_conditions.append({"전형": {"$in": real_db_values}})

            final_where = None
            if len(where_conditions) == 1:
                final_where = where_conditions[0]
            elif len(where_conditions) > 1:
                final_where = {"$and": where_conditions}

            try:
                # 검색 실행
                results = collection.query(query_texts=[prompt], n_results=5, where=final_where)
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                
                context = ""
                if docs:
                    for i, doc in enumerate(docs):
                        context += f"데이터{i+1}: [{metas[i]['학교명']} {metas[i]['전형']}] {doc}\n"
                else:
                    context = "해당 조건의 정확한 데이터가 없습니다."

                # ==========================================
                # 🔥 핵심 수정: 숫자 감각 및 로직 강화 프롬프트
                # ==========================================
                system_prompt = f"""
                당신은 냉철한 입시 분석가입니다. 아래 규칙을 절대적으로 따르세요.

                [학생 정보]
                - 내 등급: {my_grade} (숫자가 작을수록 공부 잘함)
                - 생기부: {record_level}

                [수학적 판단 규칙 (필수 준수)]
                1. 입시에서 '등급'은 1.0에 가까울수록 우수하고, 9.0에 가까울수록 저조합니다.
                2. 비교 공식: (내 등급 - 대학 커트라인) = '차이값'
                   - 차이값이 양수(+)면: 내 등급 숫자가 더 큼 -> 성적이 더 나쁨 -> **[불합격/위험/상향]**
                   - 차이값이 음수(-)면: 내 등급 숫자가 더 작음 -> 성적이 더 좋음 -> **[합격/안정/하향]**
                   - 예시: 내 등급 3.0 vs 컷 2.0 -> 차이 +1.0 (성적 부족) -> 위험
                   - 예시: 내 등급 2.0 vs 컷 3.0 -> 차이 -1.0 (성적 여유) -> 안정

                [생기부 반영 규칙]
                - 생기부가 '상/최상'이고 '학생부종합' 전형일 때만: 내 성적이 커트라인보다 0.5~0.7등급 나빠도(숫자가 커도) "소신 지원"으로 판정.
                - 그 외(교과전형, 생기부 하)는 무조건 숫자 비교만 따를 것.

                [데이터]
                {context}

                [답변 양식]
                1. **판정 결과:** (안정/소신/상향/위험 중 택1)
                2. **상세 분석:** (위 수학적 계산 결과를 근거로 설명)
                3. **조언:** (현실적인 전략 제안)
                """

                # 메모리 + 호출
                msgs = [{"role": "system", "content": system_prompt}]
                msgs.extend(st.session_state.messages[-4:])

                client = OpenAI(api_key=api_key)
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=msgs,
                    temperature=0.1 # 창의성 낮춤 (계산 정확도 위함)
                )
                answer = res.choices[0].message.content

            except Exception as e:
                answer = f"⚠️ 오류가 발생했습니다: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
