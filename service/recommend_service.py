from db import get_connection

# --- 쿼리 기반 추천 로직 ---

def recommend_by_tags(tags: list):
    if not tags:
        return []

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    format_strings = ','.join(['%s'] * len(tags))
    sql = f"""
        SELECT p.product_id, p.product_name, COUNT(*) as score
        FROM product p
        JOIN product_tag pt ON p.product_id = pt.product_id
        JOIN tag t ON pt.tag_id = t.tag_id
        WHERE t.tag_name IN ({format_strings})
        GROUP BY p.product_id
        ORDER BY score DESC
        LIMIT 8;
    """
    cursor.execute(sql, tags)
    result = cursor.fetchall()

    cursor.close()
    conn.close()
    return result

def get_trending_products(limit=12):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # 인기 상품 집계 쿼리
    query = f"""
        SELECT product_id, SUM(score) as total_score
        FROM (
            SELECT product_id, COUNT(*) * 1 AS score FROM wishlist_item GROUP BY product_id
            UNION ALL
            SELECT product_id, COUNT(*) * 3 AS score FROM order_item GROUP BY product_id
            UNION ALL
            SELECT product_id, COUNT(*) * 2 AS score FROM avatar_item GROUP BY product_id
            UNION ALL
            SELECT product_id, COUNT(*) * 2 AS score FROM closet_avatar_item GROUP BY product_id
        ) AS combined
        GROUP BY product_id
        ORDER BY total_score DESC
        LIMIT %s
    """
    cursor.execute(query, (limit,))
    popular_product_ids = cursor.fetchall()

    # 상품 정보 조회
    product_ids = [row["product_id"] for row in popular_product_ids]
    if not product_ids:
        return []

    format_strings = ','.join(['%s'] * len(product_ids))
    cursor.execute(f"""
        SELECT product_id, product_name, brand, img1
        FROM product
        WHERE product_id IN ({format_strings})
    """, product_ids)

    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return products

def get_users_by_age_range_and_gender(age_range: str, gender: str = None) -> list:
    from datetime import date
    today = date.today()

    if age_range == "10s":
        min_year, max_year = today.year - 19, today.year - 10
    elif age_range == "20s":
        min_year, max_year = today.year - 29, today.year - 20
    elif age_range == "30s":
        min_year, max_year = today.year - 39, today.year - 30
    elif age_range == "40s":
        min_year, max_year = today.year - 49, today.year - 40
    else:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT user_id FROM member WHERE birth_date BETWEEN %s AND %s"
    params = [f"{min_year}-01-01", f"{max_year}-12-31"]

    if gender and gender in ["M", "F"]:
        query += " AND gender = %s"
        params.append(gender)

    cursor.execute(query, params)
    user_ids = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    return user_ids

def get_popular_products_by_users(user_ids: list, limit: int = 12):
    if not user_ids:
        return []

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    format_str = ','.join(['%s'] * len(user_ids))
    print("format_str: ", format_str)

    # 행동 기반 상품 집계
    query = f"""
    SELECT product_id, SUM(score) AS total_score
    FROM (
        -- 위시리스트 기반 점수 (1점)
        SELECT wi.product_id, COUNT(*) * 1 AS score
        FROM wishlist_item wi
        JOIN wishlist w ON wi.wishlist_id = w.wishlist_id
        WHERE w.user_id IN ({format_str})
        GROUP BY wi.product_id

        UNION ALL

        -- 주문 기반 점수 (3점)
        SELECT oi.product_id, COUNT(*) * 3 AS score
        FROM order_item oi
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.user_id IN ({format_str})
        GROUP BY oi.product_id

        UNION ALL

        -- 아바타 try-on 기반 점수 (2점)
        SELECT ai.product_id, COUNT(*) * 2 AS score
        FROM avatar_item ai
        JOIN avatar a ON ai.avatar_id = a.avatar_id
        WHERE a.user_id IN ({format_str})
        GROUP BY ai.product_id

        UNION ALL

        -- 옷장 기반 try-on 점수 (2점)
        SELECT cai.product_id, COUNT(*) * 2 AS score
        FROM closet_avatar_item cai
        JOIN closet_avatar ca ON cai.closet_avatar_id = ca.closet_avatar_id
        WHERE ca.user_id IN ({format_str})
        GROUP BY cai.product_id
    ) AS combined
    GROUP BY product_id
    ORDER BY total_score DESC
    LIMIT %s
    """
    cursor.execute(query, user_ids * 4 + [limit])
    product_ids = [row["product_id"] for row in cursor.fetchall()]

    if not product_ids:
        return []

    format_strings = ','.join(['%s'] * len(product_ids))
    cursor.execute(f"""
        SELECT product_id, product_name, brand, img1
        FROM product
        WHERE product_id IN ({format_strings})
    """, product_ids)

    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return products


def get_similar_products(product_id: int, limit: int = 8):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # Step 1. 기준 상품의 태그 ID 리스트 추출
    cursor.execute("""
        SELECT tag_id FROM product_tag WHERE product_id = %s
    """, (product_id,))
    base_tag_ids = [row["tag_id"] for row in cursor.fetchall()]

    if not base_tag_ids:
        return []

    # Step 2. 같은 태그를 가진 다른 상품 중 공통 태그 수가 많은 상품 추출
    format_str = ','.join(['%s'] * len(base_tag_ids))
    cursor.execute(f"""
        SELECT pt.product_id, COUNT(*) as shared_tags
        FROM product_tag pt
        WHERE pt.tag_id IN ({format_str}) AND pt.product_id != %s
        GROUP BY pt.product_id
        ORDER BY shared_tags DESC
        LIMIT %s
    """, base_tag_ids + [product_id, limit])

    similar_ids = [row["product_id"] for row in cursor.fetchall()]

    if not similar_ids:
        return []

    # Step 3. 상품 정보 조회
    format_ids = ','.join(['%s'] * len(similar_ids))
    cursor.execute(f"""
        SELECT product_id, product_name, brand, img1
        FROM product
        WHERE product_id IN ({format_ids})
    """, similar_ids)

    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return products

def get_products_similar_to_user_tryon(user_id: int, limit: int = 10):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # 1. 유저가 tryon한 상품 id 목록
    cursor.execute("""
        SELECT DISTINCT ai.product_id
        FROM avatar_item ai
        JOIN avatar a ON ai.avatar_id = a.avatar_id
        WHERE a.user_id = %s
    """, (user_id,))
    tried_products = [row["product_id"] for row in cursor.fetchall()]

    if not tried_products:
        return []

    # 2. tryon 상품들의 태그 목록 수집
    format_ids = ','.join(['%s'] * len(tried_products))
    cursor.execute(f"""
        SELECT DISTINCT tag_id
        FROM product_tag
        WHERE product_id IN ({format_ids})
    """, tried_products)
    tag_ids = [row["tag_id"] for row in cursor.fetchall()]

    if not tag_ids:
        return []

    # 3. 유사 태그를 가진 다른 상품 추천
    format_tags = ','.join(['%s'] * len(tag_ids))
    cursor.execute(f"""
        SELECT pt.product_id, COUNT(*) as match_count
        FROM product_tag pt
        WHERE pt.tag_id IN ({format_tags})
          AND pt.product_id NOT IN ({format_ids})
        GROUP BY pt.product_id
        ORDER BY match_count DESC
        LIMIT %s
    """, tag_ids + tried_products + [limit])
    similar_ids = [row["product_id"] for row in cursor.fetchall()]

    if not similar_ids:
        return []

    format_similar = ','.join(['%s'] * len(similar_ids))
    cursor.execute(f"""
        SELECT product_id, product_name, brand, img1
        FROM product
        WHERE product_id IN ({format_similar})
    """, similar_ids)

    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result