from db import get_connection
from datetime import date
from collections import Counter

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

def get_user_preferred_tags(user_id: int) -> list:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    preferred_tags = set()

    # 1. 프로필 preferred_style
    cursor.execute("SELECT preferred_style FROM profile WHERE user_id = %s", (user_id,))
    row = cursor.fetchone()
    if row and row["preferred_style"]:
        preferred_tags.add(row["preferred_style"])

    # 2. member 테이블 → 성별/생년월일
    cursor.execute("SELECT birth_date, gender FROM member WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    if user:
        age = get_age_from_birth(user["birth_date"])
        preferred_tags.update(map_age_gender_to_tags(age, user["gender"]))

    # 3. 행동 기반 상품 태그 수집
    behavior_tables = ['wishlist_item', 'cart_item', 'order_item']
    product_ids = set()
    for table in behavior_tables:
        cursor.execute(f"SELECT product_id FROM {table} WHERE user_id = %s", (user_id,))
        product_ids.update(row["product_id"] for row in cursor.fetchall())

    # Tryon (avatar_item → avatar → user_id 매칭)
    cursor.execute("""
        SELECT ai.product_id FROM avatar_item ai
        JOIN avatar a ON ai.avatar_id = a.avatar_id
        WHERE a.user_id = %s
    """, (user_id,))
    product_ids.update(row["product_id"] for row in cursor.fetchall())

    # 상품 ID → 태그
    tag_counter = Counter()
    for pid in product_ids:
        cursor.execute("""
            SELECT t.name FROM tag t
            JOIN product_tag pt ON pt.tag_id = t.tag_id
            WHERE pt.product_id = %s
        """, (pid,))
        tag_counter.update(row["name"] for row in cursor.fetchall())

    for tag, _ in tag_counter.most_common(3):
        preferred_tags.add(tag)

    cursor.close()
    conn.close()
    return list(preferred_tags)