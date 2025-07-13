from db import get_connection

# 추천된 product_ids 리스트를 받아 한 번의 쿼리로 상품 정보를 ID 순서대로 반환합니다.
def get_product_details(product_ids: list) -> list:
    if not product_ids:
        return []
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # %s 개수 만큼 자리 확보
    placeholder = ','.join(['%s'] * len(product_ids))
    # FIELD()로 order by 전달 순서 유지
    sql = f"""
        SELECT product_id, product_name, brand, img1
        FROM product
        WHERE product_id IN ({placeholder})
        ORDER BY FIELD(product_id, {placeholder})
    """
    params = product_ids + product_ids
    cursor.execute(sql, params)
    results = cursor.fetchall()

    cursor.close()
    conn.close()
    return results


# product_ids 리스트에 대해 {product_id: category_name} 매핑을 반환
def get_category_map(product_ids: list[int]) -> dict[int, str]:
    if not product_ids:
        return {}
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    placeholder = ",".join(["%s"] * len(product_ids))
    cursor.execute(
        f"""
        SELECT p.product_id, c.category_name
        FROM product p
        JOIN category c ON p.category_id = c.category_id
        WHERE p.product_id IN ({placeholder})
        """,
        product_ids
    )
    mapping = {row["product_id"]: row["category_name"] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    return mapping