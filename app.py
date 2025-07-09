import os
import random
import time  # 👈 [NEW] time.sleep()을 사용하기 위해 임포트
import pandas as pd
import numpy as np
from notion_client import Client
from datetime import datetime
import nltk
from nltk.corpus import wordnet
import streamlit as st
from st_click_detector import click_detector
from gtts import gTTS
from io import BytesIO

# --- 0. 페이지 설정 및 NLTK 데이터 다운로드 ---
st.set_page_config(page_title="VOCA Master", page_icon="📚", layout="centered")

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
download_nltk_data()

# --- 1. 데이터 로딩 및 전처리, TTS 함수 ---

@st.cache_data(ttl=600)
def get_db_as_dataframe(database_id: str, token: str) -> pd.DataFrame | None:
    notion = Client(auth=token)
    try:
        response = notion.databases.query(database_id=database_id)
        pages = response.get('results', [])
        while response.get('has_more'):
            cursor = response.get('next_cursor')
            response = notion.databases.query(database_id=database_id, start_cursor=cursor)
            pages.extend(response.get('results', []))

        processed_pages = []
        for page in pages:
            props = page.get('properties', {})
            word_prop = props.get('word', {}).get('title', [])
            synonyms_prop = props.get('Synonyms', {}).get('rich_text', [])

            processed_pages.append({
                'last_edited_time': page.get('last_edited_time'),
                'word': word_prop[0]['plain_text'] if word_prop else None,
                'Synonyms': synonyms_prop[0]['plain_text'] if synonyms_prop else None,
            })

        if not processed_pages: return pd.DataFrame()

        df = pd.DataFrame(processed_pages)
        if 'last_edited_time' in df.columns:
            df['last_edited_time'] = pd.to_datetime(df['last_edited_time'], errors='coerce')
        return df

    except Exception as e:
        st.error(f"❌ Notion 데이터 로딩 오류: {e}")
        return None

@st.cache_data
def create_synonym_groups(df: pd.DataFrame, decay_const: float = 0.023) -> list[dict]:
    if df is None or not all(col in df.columns for col in ['word', 'Synonyms', 'last_edited_time']):
        return []

    structured_groups = []
    now = pd.Timestamp.now(tz='Asia/Seoul')
    for _, row in df.iterrows():
        main_word, synonyms_str = row['word'], row['Synonyms']
        timestamp = row['last_edited_time']

        if not (isinstance(main_word, str) and main_word.strip()): continue

        synonyms_list = [s.strip() for s in synonyms_str.split(',') if s.strip()] if isinstance(synonyms_str, str) else []
        if synonyms_list:
            weight = 1.0
            if pd.notna(timestamp):
                days_elapsed = (now - timestamp).total_seconds() / (24 * 3600)
                weight = np.exp(-decay_const * days_elapsed)

            structured_groups.append({'main': main_word.strip(), 'synonyms': synonyms_list, 'weight': weight})

    return structured_groups

def get_synset(word):
    try:
        return wordnet.synsets(word)[0]
    except IndexError:
        return None

@st.cache_data
def text_to_speech(text: str) -> bytes | None:
    try:
        tts = gTTS(text=text, lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.getvalue()
    except Exception as e:
        st.error(f"음성 변환 중 오류 발생: {e}")
        return None

# --- 2. 퀴즈 문제 생성 로직 (기존과 동일) ---
def generate_quiz_questions(groups: list[dict], num_questions: int, similarity_threshold=0.6):
    questions = []
    if not groups: return []
    weights = [g['weight'] for g in groups]
    all_words = list(set(w for group in groups for w in [group['main']] + group['synonyms']))
    used_question_words = set()
    max_attempts = num_questions * 5
    attempts = 0

    while len(questions) < num_questions and attempts < max_attempts:
        attempts += 1
        correct_group = random.choices(groups, weights=weights, k=1)[0]
        if random.random() < 0.8:
            question_word, answer_word = correct_group['main'], random.choice(correct_group['synonyms'])
        else:
            question_word, answer_word = random.choice(correct_group['synonyms']), correct_group['main']
        if question_word in used_question_words: continue
        question_synset = get_synset(question_word)
        if not question_synset: continue
        distractors = []
        candidate_pool = [w for w in all_words if w not in correct_group['main'] and w not in correct_group['synonyms']]
        random.shuffle(candidate_pool)
        for candidate in candidate_pool:
            if len(distractors) == 3: break
            candidate_synset = get_synset(candidate)
            if candidate_synset:
                similarity = question_synset.wup_similarity(candidate_synset)
                if similarity is not None and similarity < similarity_threshold:
                    distractors.append(candidate)
        if len(distractors) < 3: continue
        used_question_words.add(question_word)
        options = [answer_word] + distractors
        random.shuffle(options)
        options.append("I don't know.")
        questions.append({
            "question_word": question_word,
            "options": options,
            "answer": answer_word
        })
    return questions

# --- 3. Streamlit UI 구성 ---

NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
DATABASE_ID = st.secrets["DATABASE_ID"]

st.sidebar.title("MENU")
app_mode = st.sidebar.radio(
    "모드를 선택하세요",
    ["📖 암기 모드 (Study Mode)", "✍️ 퀴즈 모드 (Quiz Mode)"]
)

if not (NOTION_TOKEN and DATABASE_ID):
    st.error("`.env` 파일에 `NOTION_TOKEN`과 `DB_ID`를 설정해주세요.")
    st.stop()

df = get_db_as_dataframe(DATABASE_ID, NOTION_TOKEN)
synonym_groups = create_synonym_groups(df)

if df is None or df.empty:
    st.warning("Notion에서 데이터를 불러오지 못했거나 데이터베이스가 비어 있습니다.")
    st.stop()

# --- 퀴즈 모드 (기존과 동일) ---
if app_mode == "✍️ 퀴즈 모드 (Quiz Mode)":
    # ... (퀴즈 모드 코드는 변경 없음) ...
    st.title("✍️ TOEFL VOCA TEST")

    if 'test_started' not in st.session_state:
        st.session_state.test_started = False

    if st.session_state.test_started:
        current_q_index = st.session_state.current_q
        total_questions = len(st.session_state.questions)
        if current_q_index < total_questions:
            st.progress((current_q_index + 1) / total_questions, text=f"Question {current_q_index + 1}/{total_questions}")
            question_data = st.session_state.questions[current_q_index]
            st.subheader(f"Q: '{question_data['question_word']}'의 유의어는?")
            user_choice = st.radio("다음 중 정답을 고르세요:", options=question_data['options'], index=None, key=f"q_{current_q_index}")

            if st.button("확인", key=f"submit_{current_q_index}"):
                if user_choice:
                    st.session_state.user_answers[current_q_index] = user_choice
                    if user_choice == question_data['answer']:
                        st.session_state.score += 1
                        st.success(f"✅ 정답입니다!", icon="🎉")
                    else:
                        st.error(f"❌ 오답입니다. 정답은 '{question_data['answer']}'입니다.", icon="🤔")
                    st.session_state.current_q += 1
                    st.rerun()
                else:
                    st.warning("답을 선택해주세요!")
        else:
            st.header("✨ 테스트 결과")
            score = st.session_state.score
            st.metric(label="정답률", value=f"{score / total_questions:.2%}", delta=f"{score} / {total_questions} 문제")
            st.balloons()

            if not st.session_state.get('result_saved', False):
                incorrect_answers = []
                for i, q_data in enumerate(st.session_state.questions):
                    user_answer = st.session_state.user_answers.get(i, "N/A")
                    if user_answer != q_data['answer']:
                        incorrect_answers.append({
                            "문제 번호": i + 1, "문제 단어": q_data['question_word'],
                            "선택한 답": user_answer, "정답": q_data['answer']
                        })
                if incorrect_answers:
                    result_df = pd.DataFrame(incorrect_answers)
                    result_dir = "result"
                    os.makedirs(result_dir, exist_ok=True)
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"incorrect_answers_{timestamp_str}_{total_questions}q.txt"
                    filepath = os.path.join(result_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write("--- 오답 노트 ---\n\n")
                        f.write(result_df.to_string(index=False))
                    st.success(f"틀린 문제가 '{filepath}'에 저장되었습니다.")
                else:
                    st.success("🎉 모든 문제를 맞혔습니다! 저장할 오답이 없습니다.")
                st.session_state.result_saved = True

            st.subheader("📝 문제 다시보기")
            for i, q_data in enumerate(st.session_state.questions):
                with st.expander(f"{'✅' if st.session_state.user_answers.get(i) == q_data['answer'] else '❌'} Q{i+1}. '{q_data['question_word']}'"):
                    st.markdown(f"**- 선택한 답:** `{st.session_state.user_answers.get(i, '답변 안 함')}`")
                    if st.session_state.user_answers.get(i) != q_data['answer']:
                        st.markdown(f"**- 정답:** `{q_data['answer']}`")
            if st.button("새로운 퀴즈 시작하기"):
                st.session_state.test_started = False
                st.session_state.pop('questions', None)
                st.session_state.pop('result_saved', None)
                st.rerun()

    else:
        st.header("⚙️ 테스트 설정")
        max_q = len(synonym_groups)
        num_q_input = st.number_input(
            "풀고 싶은 문제 수를 입력하세요:", min_value=5, max_value=max_q,
            value=min(25, max_q), step=1
        )
        similarity_threshold = st.slider(
            "오답 선택지 난이도 조절 (Similarity Threshold):",
            min_value=0.1, max_value=0.9, value=0.6, step=0.05
        )
        st.info("""
            **난이도 설명:** 이 값이 오답과 문제 단어의 의미적 유사도 임계값입니다.
            - **값이 낮을수록 (Easy):** 오답이 문제와 관련 없는 단어로 구성되어 쉬워집니다.
            - **값이 높을수록 (Hard):** 오답이 문제와 의미적으로 유사한 단어로 구성되어 어려워집니다.
        """)
        if st.button("퀴즈 시작하기!", type="primary"):
            with st.spinner("유사도 기반으로 문제를 생성하는 중입니다..."):
                questions = generate_quiz_questions(synonym_groups, num_q_input, similarity_threshold)

            if len(questions) >= num_q_input:
                st.session_state.questions = questions
                st.session_state.current_q = 0
                st.session_state.score = 0
                st.session_state.user_answers = {}
                st.session_state.test_started = True
                st.session_state.result_saved = False
                st.rerun()
            else:
                st.error(f"요청하신 {num_q_input}개의 문제를 생성하지 못했습니다 (생성된 문제: {len(questions)}개). Notion DB의 단어 수를 늘리거나 난이도를 낮춰보세요.")


# --- 암기 모드 ---
elif app_mode == "📖 암기 모드 (Study Mode)":
    st.title("📖 TOEFL VOCA 암기장 (Flashcard Mode)")
    st.info(f"총 {len(synonym_groups)}개의 단어가 있습니다. 카드를 클릭하면 유의어를 확인할 수 있습니다. 🖱️")

    # 👈 [MODIFIED] 세션 상태 초기화 강화
    if 'study_groups' not in st.session_state:
        st.session_state.study_groups = random.sample(synonym_groups, len(synonym_groups))
        st.session_state.card_index = 0
        st.session_state.card_flipped = False
        st.session_state.is_playing = False # 재생 상태 추가

    if not st.session_state.study_groups:
        st.warning("학습할 단어가 없습니다.")
        st.stop()

    # 👈 [NEW] 재생 중지 함수: 다른 버튼 클릭 시 재생을 멈추기 위함
    def stop_playing():
        st.session_state.is_playing = False

    total_cards = len(st.session_state.study_groups)
    current_index = st.session_state.card_index

    col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 2.0, 5, 1.5])

    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            stop_playing() # 재생 중지
            if current_index > 0:
                st.session_state.card_index -= 1
                st.session_state.card_flipped = False
                st.rerun()

    with col2:
        if st.button("다음 ➡️", use_container_width=True):
            stop_playing() # 재생 중지
            if current_index < total_cards - 1:
                st.session_state.card_index += 1
                st.session_state.card_flipped = False
                st.rerun()

    # 👈 [MODIFIED] 재생/일시정지 토글 버튼 로직
    with col3:
        # 재생 중일 때는 '일시정지' 버튼을, 아닐 때는 '전체 재생' 버튼을 보여줌
        if st.session_state.is_playing:
            if st.button("⏸️ 일시정지", use_container_width=True):
                stop_playing()
                st.rerun()
        else:
            if st.button("▶️ 전체 재생", use_container_width=True):
                st.session_state.is_playing = True
                st.rerun()

    with col4:
        # 재생 중일 때는 어떤 단어를 읽고 있는지 표시
        progress_text = f"Playing: {st.session_state.study_groups[current_index]['main']}" if st.session_state.is_playing else f"Card {current_index + 1} / {total_cards}"
        st.progress((current_index + 1) / total_cards, text=progress_text)

    with col5:
        if st.button("🔄 셔플", use_container_width=True):
            stop_playing() # 재생 중지
            st.session_state.study_groups = random.sample(synonym_groups, len(synonym_groups))
            st.session_state.card_index = 0
            st.session_state.card_flipped = False
            st.rerun()

    st.divider()

    # 오디오 플레이어를 위한 보이지 않는 공간
    audio_placeholder = st.empty()

    # --- 카드 UI (기존과 동일) ---
    current_group = st.session_state.study_groups[current_index]
    main_word = current_group['main']
    synonyms = current_group['synonyms']
    card_css = """
    <style>
        .card-container { width: 100%; height: 250px; perspective: 1000px; }
        .card-flipper { width: 100%; height: 100%; position: relative; transform-style: preserve-3d; transition: transform 0.6s; }
        .card-flipper.is-flipped { transform: rotateY(180deg); }
        .card-face { position: absolute; width: 100%; height: 100%; -webkit-backface-visibility: hidden; backface-visibility: hidden; display: flex; align-items: center; justify-content: center; border: 1px solid #e6e6e6; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .card-back { transform: rotateY(180deg); align-items: flex-start; padding-top: 20px; }
    </style>
    """
    synonyms_html_list = "".join(f"<li style='text-align: left; margin: 5px 0;'><code style='font-size: 1.1rem;'>{s}</code></li>" for s in synonyms)
    flip_class = "is-flipped" if st.session_state.card_flipped else ""

    html_content = f"""
    {card_css}
    <a href='#' id='card-link-{current_index}' style='text-decoration: none; color: inherit;'>
        <div class="card-container">
            <div class="card-flipper {flip_class}">
                <div class="card-face card-front"><h1 style='color: steelblue;'>{main_word}</h1></div>
                <div class.card-face card-back">
                    <div style='height: 100%; width: 80%; overflow-y: auto;'>
                        <ul style='list-style-position: inside; padding-left: 10%;'>{synonyms_html_list}</ul>
                    </div>
                </div>
            </div>
        </div>
    </a>
    """
    # 재생 중에는 카드 클릭(뒤집기) 방지
    if not st.session_state.is_playing:
        clicked = click_detector(html_content, key=f"detector_{current_index}")
        if clicked:
            st.session_state.card_flipped = not st.session_state.card_flipped
            st.rerun()
    else:
        # 재생 중일 때는 클릭 감지기 없이 HTML만 렌더링
        st.markdown(html_content, unsafe_allow_html=True)


    # 👈 [NEW] 전체 순회 재생을 위한 핵심 로직
    # 이 로직은 스크립트의 맨 마지막에 위치하여 다른 UI가 모두 렌더링된 후 실행됩니다.
    if st.session_state.is_playing:
        # 현재 인덱스가 유효한 범위 내에 있는지 확인
        if current_index < total_cards:
            # 읽을 텍스트 생성
            text_to_read = f"{main_word}. Synonyms are: {', '.join(synonyms)}"
            
            # 음성 데이터 생성
            audio_bytes = text_to_speech(text_to_read)

            if audio_bytes:
                # 보이지 않는 플레이스홀더에 자동 재생 오디오를 렌더링
                audio_placeholder.audio(audio_bytes, format="audio/mp3", autoplay=True)
                
                # 다음 단어로 인덱스 이동
                st.session_state.card_index += 1
                
                # 오디오가 재생될 시간을 주기 위해 짧게 대기
                # (이 값을 조절하여 재생 속도를 바꿀 수 있습니다)
                time.sleep(1.5)

                # 다음 단어 재생을 위해 스크립트를 다시 실행
                st.rerun()
            else:
                # 음성 생성 실패 시 재생 중지
                st.error("음성 파일을 생성하지 못했습니다. 재생을 중지합니다.")
                stop_playing()
                st.rerun()
        else:
            # 모든 단어 재생 완료 시
            st.toast("✅ 모든 단어 학습을 완료했습니다!", icon="🎉")
            st.session_state.card_index = 0  # 인덱스를 처음으로 리셋
            stop_playing()
            st.rerun()