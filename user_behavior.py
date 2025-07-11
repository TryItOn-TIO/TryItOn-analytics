from db import get_connection

# recommend_behavior_log 테이블에서 유저의 로그를 조회하고, product_tag → tag를 JOIN하여 태그별 score 합계를 구합니다.
def get_user_behavior_logs(user_id: int) -> list[tuple[str, float]]:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT t.tag_name, SUM(r.score) AS total_score
        FROM recommend_behavior_log r
        JOIN product_tag pt ON r.product_id = pt.product_id
        JOIN tag t ON pt.tag_id = t.tag_id
        WHERE r.user_id = %s
        GROUP BY t.tag_name
    """, (user_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    # (tag, score) 리스트로 반환
    return [(row["tag_name"], row["total_score"]) for row in rows]