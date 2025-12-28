import streamlit as st
import chromadb
from openai import OpenAI
import os
from chromadb.utils import embedding_functions

# ==========================================
# 1. 기본 설정 및 DB 연결
# ==========================================
st.set_page_config(page_title="입시 컨설팅 AI", page_icon="🎓", layout="wide")
st.title("🎓 대입 합격예측 AI 컨설턴트")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")

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
# 2. (핵심 수정) 스마트 필터링 로직
# ==========================================
@st.cache_data
def get_filter_options():
    """
    DB에서 학교명과 전형을 가져와서
    띄어쓰기가 달라도 같은 의미면 하나로 합칩니다.
    """
    try:
        data = collection.get(include=["metadatas"])
        
        # 학교명 처리
        school_set = set()
        
        # 전형 처리 (매핑 딕셔너리 생성)
        # 예: {'학생부교과전형': ['학생부 교과 전형', '학생부 교과전형']}
        type_map = {} 
        
        for meta in data['metadatas']:
            # 학교명 수집
            if "학교명" in meta and meta["학교명"]:
                school_set.add(meta["학교명"])
            
            # 전형 이름 정규화 (띄어쓰기 제거)
            if "전형" in meta and meta["전형"]:
                raw_val = meta["전형"]
                # 띄어쓰기를 모두 없앤 이름을 '대표 이름'으로 사용
                clean_name = raw_val.replace(" ", "")
                
                if clean_name not in type_map:
                    type_map[clean_name] = []
                # 실제 DB에 있는 값을 리스트에 추가 (나중에 검색할 때 씀)
                if raw_val not in type_map[clean_name]:
                    type_map[clean_name].append(raw_val)

        return sorted(list(school_set)), type_map
    except:
        return [], {}

# 학교 목록과 전형 매핑 정보 가져오기
school_list, type_mapping = get_filter_options()

# 사이드바 표시용 전형 리스트 (띄어쓰기 없는 깔끔한 이름들)
display_types = ["전체"] + sorted(list(type_mapping.keys()))

# ==========================================
# 3. 사이드바 UI
# ==========================================
st.sidebar.header("📝 학생 정보 입력")

target_school = st.sidebar.selectbox("희망 대학", ["전체"] + school_list)
# 사용자는 깔끔한 이름("학생부교과전형")을 선택함
selected_display_type = st.sidebar.selectbox("희망 전형", display_types)

my_grade = st.sidebar.number_input(
    "내신 등급 (직접 입력)", 
    min_value=1.00, max_value=9.00, value=3.00, step=0.00, format="%.2f"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 생활기록부 수준")
record_level = st.sidebar.select_slider(
    "생기부 퀄리티 선택",
    options=["하 (기본)", "중 (평범)", "상 (우수)", "최상 (특목고)"],
    value="중 (평범)"
)

# ==========================================
# 4. RAG 및 대화 로직
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 어떤 대학/학과를 목표로 하시나요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("질문 입력"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            
            # --- 1. 필터 조건 생성 (고급) ---
            where_conditions = []
            
            # 학교 필터
            if target_school != "전체":
                where_conditions.append({"학교명": target_school})
            
            # 전형 필터 (핵심 수정!)
            if selected_display_type != "전체":
                # 사용자가 선택한 '깔끔한 이름'에 연결된 '실제 DB 값들'을 모두 가져옴
                # 예: ["학생부 교과 전형", "학생부 교과전형"]
                real_db_values = type_mapping[selected_display_type]
                
                if len(real_db_values) == 1:
                    # 값이 하나면 단순 일치 검색
                    where_conditions.append({"전형": real_db_values[0]})
                else:
                    # 값이 여러 개면 $in 연산자로 "이거 아니면 저거" 검색
                    where_conditions.append({"전형": {"$in": real_db_values}})

            # ChromaDB where 절 조합
            final_where = None
            if len(where_conditions) == 1:
                final_where = where_conditions[0]
            elif len(where_conditions) > 1:
                final_where = {"$and": where_conditions}

            # --- 2. 검색 및 답변 ---
            try:
                results = collection.query(
                    query_texts=[prompt],
                    n_results=5,
                    where=final_where
                )
                
                docs = results['documents'][0]
                metas = results['metadatas'][0]
                
                context = ""
                if docs:
                    for i, doc in enumerate(docs):
                        # 전형 이름을 보여줄 때도 깔끔하게 표시 가능
                        context += f"[{metas[i]['학교명']} {metas[i]['전형']}] {doc}\n"
                else:
                    context = "조건에 맞는 데이터가 없습니다. 일반적인 입시 조언을 제공합니다."

                # 시스템 프롬프트 (생기부 로직 반영)
                system_prompt = f"""
                당신은 입시 컨설턴트입니다.
                
                [학생 정보]
                - 내신: {my_grade}등급
                - 생기부: {record_level}
                
                [판단 로직]
                1. 생기부 '상/최상': 학종 지원 시 내신 컷보다 0.5~0.8 낮아도 '소신/적정' 판정.
                2. 생기부 '중/하': 학종보다는 교과 위주 추천. 내신 컷 준수 필수.
                3. 데이터의 '50% cut', '70% cut'과 학생 내신을 비교하여 합격 확률(%)을 추정하세요.

                [입시 데이터]
                {context}
                
                위 정보를 바탕으로 이전 대화 맥락을 고려하여 답변하세요.
                """
                
                # 메모리 기능
                msgs = [{"role": "system", "content": system_prompt}]
                msgs.extend(st.session_state.messages[-4:]) # 최근 4개 대화 기억

                client = OpenAI(api_key=api_key)
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=msgs,
                    temperature=0.2
                )
                answer = res.choices[0].message.content

            except Exception as e:
                answer = f"오류 발생: {e}"

            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})