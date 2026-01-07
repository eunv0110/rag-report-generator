"""RAG 보고서 생성 Streamlit 애플리케이션"""
import streamlit as st
from streamlit_chat import message
import requests
import json
from datetime import datetime, timedelta
import time

# 페이지 설정
st.set_page_config(
    page_title="보고서 자동생성 서비스",
    page_icon="👩🏻‍💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 깔끔한 단색 스타일 설정
st.markdown("""
    <style>
    /* 전역 스타일 */
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }

    /* 헤더 스타일 */
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }

    /* 버튼 스타일 */
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* 다운로드 버튼 스타일 */
    .stDownloadButton>button {
        background-color: #9b59b6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .stDownloadButton>button:hover {
        background-color: #8e44ad;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* 입력 필드 스타일 */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #dce1e6;
        padding: 0.75rem;
        font-size: 1rem;
    }

    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.1);
    }

    /* 채팅 컨테이너 스타일 */
    .chat-container {
        background-color: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        min-height: 500px;
    }

    /* 상태 카드 스타일 */
    .status-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* 진행 상황 말풍선 스타일 */
    .progress-bubble {
        background-color: #3498db;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* 성공 메시지 스타일 */
    .success-bubble {
        background-color: #27ae60;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    <script>
    // 페이지 로드 시 스크롤을 맨 아래로 이동
    window.addEventListener('load', function() {
        window.scrollTo(0, document.body.scrollHeight);
    });
    // DOM 변경 감지하여 새 메시지 추가 시 스크롤
    var observer = new MutationObserver(function() {
        window.scrollTo(0, document.body.scrollHeight);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
""", unsafe_allow_html=True)

# API 엔드포인트 설정
API_BASE_URL = "http://localhost:8000"

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'progress_messages' not in st.session_state:
    st.session_state.progress_messages = []
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'pending_request' not in st.session_state:
    st.session_state.pending_request = None

# 사이드바 - 서버 상태와 설정
with st.sidebar:
    st.markdown("###👩🏻‍💻 보고서 자동생성 서비스")
    st.markdown("---")

    # 서버 상태 체크
    st.markdown("#### 서버 상태")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.markdown("""
                <div class="status-card" style="background-color: #27ae60; color: white;">
                    <strong>✅ 연결됨</strong>
                </div>
            """, unsafe_allow_html=True)
            health_data = response.json()
        else:
            st.markdown("""
                <div class="status-card" style="background-color: #e74c3c; color: white;">
                    <strong>❌ 응답 오류</strong>
                </div>
            """, unsafe_allow_html=True)
    except requests.exceptions.RequestException:
        st.markdown("""
            <div class="status-card" style="background-color: #e74c3c; color: white;">
                <strong>❌ 연결 실패</strong>
            </div>
        """, unsafe_allow_html=True)
        st.caption("서버를 시작하세요: `python run_api.py`")

    # 설정 섹션
    st.markdown("#### 보고서 설정")

    default_author = st.text_input(
        "작성자",
        placeholder="이름을 입력하세요",
        help="보고서 작성자 이름"
    )

    report_type_option = st.selectbox(
        "보고서 타입",
        options=["weekly", "executive"],
        format_func=lambda x: {"weekly": "📅 주간 보고서", "executive": "👔 최종 보고서"}[x],
        help="생성할 보고서 타입을 선택하세요"
    )


    default_output_format = st.selectbox(
        "출력 형식",
        options=["pdf", "docx"],
        format_func=lambda x: {"pdf": "📕 PDF", "docx": "📘 DOCX"}[x],
        help="보고서 출력 형식을 선택하세요"
    )

    use_date_filter = st.checkbox(
        "날짜 범위 필터링",
        value=False,
        help="특정 날짜 범위의 문서만 검색합니다"
    )

    if use_date_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_date_input = st.date_input(
                "시작 날짜",
                value=datetime.now() - timedelta(days=7),
                help="검색 시작 날짜"
            )
        with col2:
            end_date_input = st.date_input(
                "종료 날짜",
                value=datetime.now(),
                help="검색 종료 날짜"
            )
        default_start_date = start_date_input.strftime("%Y-%m-%d")
        default_end_date = end_date_input.strftime("%Y-%m-%d")
    else:
        default_start_date = None
        default_end_date = None

    st.markdown("---")

    # 채팅 기록 삭제
    if st.button("🗑️ 채팅 기록 삭제", use_container_width=True):
        st.session_state.messages = []
        st.session_state.progress_messages = []
        st.rerun()

    st.caption("💡 쉽게 보고서를 만들어보세요!")


# 메인 컨텐츠
st.title("👩🏻‍💻 보고서 자동생성 서비스")
st.markdown('<p style="font-size: 1.1rem; color: #64748b;">보고서를 요청하면 AI가 자동으로 생성해드립니다.</p>', unsafe_allow_html=True)


# 채팅 메시지 컨테이너
chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) == 0 and not st.session_state.is_generating:
        st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #94a3b8;">
                <h3>👋 안녕하세요!</h3>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    아래에 원하시는 보고서를 요청해주세요.<br>
                    예: "이번 주 주간 보고서 작성해줘"
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{i}", avatar_style="big-smile", seed=456)
            elif msg["role"] == "assistant":
                message(msg["content"], is_user=False, key=f"bot_{i}", avatar_style="bottts", seed=789)

                # 파일 다운로드 버튼이 있는 경우
                if "file_data" in msg:
                    st.download_button(
                        label=f"📥 {msg['filename']} 다운로드",
                        data=msg["file_data"],
                        file_name=msg["filename"],
                        mime=msg["mime_type"],
                        use_container_width=True,
                        key=f"download_{i}"
                    )

# 진행 상황 말풍선 표시
if st.session_state.progress_messages:
    st.markdown("### 🔄 진행 상황")
    for prog_msg in st.session_state.progress_messages:
        st.markdown(f"""
            <div class="progress-bubble">
                {prog_msg}
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 입력 영역 (전체 너비로 하단에 배치)
user_input = st.chat_input("예: 이번 주 주간 보고서 작성해줘")

# 메시지 전송 처리
if user_input and not st.session_state.pending_request:
    # 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })

    # 요청 데이터 구성
    request_data = {
        "report_type": report_type_option,
        "output_format": default_output_format,
        "question": user_input
    }

    if default_author:
        request_data["author"] = default_author

    if default_start_date:
        request_data["start_date"] = default_start_date

    if default_end_date:
        request_data["end_date"] = default_end_date

    # 요청 정보를 세션에 저장하고 페이지 새로고침
    st.session_state.pending_request = request_data
    st.session_state.is_generating = True
    st.rerun()

# 대기 중인 요청이 있으면 처리
if st.session_state.pending_request and st.session_state.is_generating:
    request_data = st.session_state.pending_request

    # 진행 상황 초기화
    st.session_state.progress_messages = []

    # 진행 상황을 표시하는 컨테이너 생성
    progress_container = st.container()

    with progress_container:
        # 진행 상황 말풍선 표시 (채팅 메시지 형식)
        progress_messages_temp = []

        try:
            # 1단계: 요청 초기화
            progress_msg_1 = st.empty()
            with progress_msg_1:
                message("🚀 보고서 생성 요청을 시작합니다...", is_user=False, key="progress_1", avatar_style="bottts", seed=789)
            time.sleep(0.5)

            # 2단계: 데이터 검색
            progress_msg_2 = st.empty()
            with progress_msg_2:
                message("🔍 관련 문서를 검색하고 있습니다...", is_user=False, key="progress_2", avatar_style="bottts", seed=789)
            time.sleep(0.5)

            # 3단계: AI 분석
            progress_msg_3 = st.empty()
            with progress_msg_3:
                message("🤖 AI가 문서를 분석하고 있습니다...", is_user=False, key="progress_3", avatar_style="bottts", seed=789)

            start_time = time.time()

            # API 호출
            response = requests.post(
                f"{API_BASE_URL}/api/v1/generate-report",
                json=request_data,
                timeout=300
            )

            generation_time = time.time() - start_time

            # 4단계: 보고서 작성
            progress_msg_4 = st.empty()
            with progress_msg_4:
                message("📝 보고서를 작성하고 있습니다...", is_user=False, key="progress_4", avatar_style="bottts", seed=789)
            time.sleep(0.3)

            if response.status_code == 200:
                # 5단계: 완료
                progress_msg_5 = st.empty()
                with progress_msg_5:
                    message("✅ 보고서 생성이 완료되었습니다!", is_user=False, key="progress_5", avatar_style="bottts", seed=789)
                time.sleep(0.5)

                # 파일명 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_type = request_data["report_type"]
                output_format = request_data["output_format"]
                filename = f"{report_type}_report_{timestamp}.{output_format}"

                # 어시스턴트 메시지 추가 (간단하게)
                assistant_message = f"✅ 보고서가 성공적으로 생성되었습니다! ({generation_time:.2f}초)"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message,
                    "timestamp": datetime.now().isoformat(),
                    "file_data": response.content,
                    "filename": filename,
                    "mime_type": "application/pdf" if output_format == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                })

            else:
                # 에러 응답
                error_message = f"❌ 오류가 발생했습니다 (상태 코드: {response.status_code})"
                try:
                    error_data = response.json()
                    error_message += f"\n\n상세 내용: {error_data.get('detail', response.text)}"
                except:
                    error_message += f"\n\n상세 내용: {response.text}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": datetime.now().isoformat()
                })

        except requests.exceptions.Timeout:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ 요청 시간이 초과되었습니다. 날짜 범위를 줄이거나 나중에 다시 시도해주세요.",
                "timestamp": datetime.now().isoformat()
            })

        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.",
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ 예기치 않은 오류가 발생했습니다: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })

        finally:
            # 진행 상황 초기화
            st.session_state.progress_messages = []
            st.session_state.is_generating = False
            st.session_state.pending_request = None

    # 페이지 새로고침
    st.rerun()

