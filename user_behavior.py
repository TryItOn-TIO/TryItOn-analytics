from db import get_connection

# recommend_behavior_log 테이블에서 유저의 로그를 조회하고, product_tag → tag를 JOIN하여 태그별 score 합계를 구합니다.
def get_user_behavior_logs(user_id: int) -> list[tuple[str, float]]:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # 태그별 점수 집계
    cursor.execute("""
        SELECT t.tag_name AS name, SUM(r.score) AS total_score
        FROM recommend_behavior_log r
        JOIN product_tag pt ON r.product_id = pt.product_id
        JOIN tag t ON pt.tag_id = t.tag_id
        WHERE r.user_id = %s
        GROUP BY t.tag_name
    """, (user_id,))
    tag_rows = cursor.fetchall()

    # 카테고리별 점수 집계
    cursor.execute("""
        SELECT c.category_name AS name, SUM(r.score) AS total_score
        FROM recommend_behavior_log r
        JOIN product p ON r.product_id = p.product_id
        JOIN category c ON p.category_id = c.category_id
        WHERE r.user_id = %s
        GROUP BY c.category_name
    """, (user_id,))
    cat_rows = cursor.fetchall()

    cursor.close()
    conn.close()

    # 합쳐서 (이름, 점수) 리스트로 반환
    combined = [(row["name"], row["total_score"]) for row in tag_rows]
    combined += [(row["name"], row["total_score"]) for row in cat_rows]
    return combined