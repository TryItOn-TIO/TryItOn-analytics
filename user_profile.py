from db import get_connection
from datetime import date

def get_age_from_birth(birth_date):
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

def map_age_gender_to_tags(age: int, gender: str):
    tags = []
    if age < 20:
        tags.append("10대추천")
    elif age < 30:
        tags.append("20대추천")
    elif age < 40:
        tags.append("30대추천")
    if gender == "M":
        tags.append("남성추천")
    elif gender == "F":
        tags.append("여성추천")
    else:
        tags.append("유니섹스")
    return tags

# 선호하는 스타일, 성별, 나이를 고려하여 유저가 선호할 만한 태그 전체를 반환한다.
def get_user_preferred_tags(user_id: int) -> list:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    preferred_tags = []

    # 1. 프로필 preferred_style
    cursor.execute("SELECT preferred_style FROM profile WHERE member_user_id = %s", (user_id,))
    row = cursor.fetchone()
    if row and row["preferred_style"]:
        if row["preferred_style"] not in preferred_tags:
            preferred_tags.append(row["preferred_style"])

    # 2. member 테이블 → 성별/생년월일
    cursor.execute("SELECT birth_date, gender FROM member WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    if user:
        age = get_age_from_birth(user["birth_date"])
        for tag in map_age_gender_to_tags(age, user["gender"]):
            if tag not in preferred_tags:
                preferred_tags.append(tag)

    cursor.close()
    conn.close()
    return preferred_tags