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
    cursor.execute("SELECT preferred_style FROM profile WHERE member_user_id = %s", (user_id,))
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
    behavior_queries = {
    "wishlist_item": """
        SELECT wi.product_id
        FROM wishlist_item wi
        JOIN wishlist w ON wi.wishlist_id = w.wishlist_id
        WHERE w.user_id = %s
    """,
    "cart_item": """
        SELECT pv.product_id
        FROM product_variant pv
        JOIN cart_item ci ON ci.variant_id = pv.variant_id
        JOIN cart c ON ci.cart_id = c.cart_id
        WHERE c.user_id = %s
    """,
    "order_item": """
        SELECT oi.product_id
        FROM order_item oi
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.user_id = %s
    """,
    "avatar_item": """
        SELECT ai.product_id
        FROM avatar_item ai
        JOIN avatar a ON ai.avatar_id = a.avatar_id
        WHERE a.user_id = %s
    """
    }

    product_ids = set()
    for label, query in behavior_queries.items():
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()
        product_ids.update(row.get("product_id") for row in rows if row.get("product_id") is not None)

    # 상품 ID → 태그
    tag_counter = Counter()
    for pid in product_ids:
        cursor.execute("""
            SELECT t.tag_name FROM tag t
            JOIN product_tag pt ON pt.tag_id = t.tag_id
            WHERE pt.product_id = %s
        """, (pid,))
        tag_counter.update(row["tag_name"] for row in cursor.fetchall())

    for tag, _ in tag_counter.most_common(3):
        preferred_tags.add(tag)

    cursor.close()
    conn.close()
    return list(preferred_tags)