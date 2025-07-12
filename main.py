from fastapi import FastAPI, Query
from user_profile import get_user_preferred_tags
from repository import get_product_details, get_category_map
from service.recommend_service import *
from service.recommend_vector_service import VectorRecommender, diversify_by_category

app = FastAPI()
recommender = VectorRecommender()

@app.get("/")
async def root():
    return {"message": "Hello TIO"}

# ---------------- 벡터 기반 ----------------
@app.get("/recommend/hybrid")
def hybrid(user_id: int = Query(...), top_n: int = Query(8)):
    # 벡터 기반 추천으로 3배수의 상품 뽑기
    candidate_ids = recommender.recommend_by_user(user_id, top_n * 3)

    # 한 번에 카테고리 매핑 로드
    cat_map = get_category_map(candidate_ids)

    # 카테고리별 4개씩, 총 top_n개로 다양화
    diversified = diversify_by_category(candidate_ids, cat_map, max_per_cat=4, top_n=top_n)

    # 최종 상품 정보 조회
    products = get_product_details(diversified)
    return {"user_id": user_id, "recommendations": products}

# ---------------- 쿼리 기반 ----------------
# 유저의 개인화 추천 상품 반환
@app.get("/recommend/for-you")
def recommend(user_id: int = Query(..., description="로그인한 유저의 ID")):
    preferred_tags = get_user_preferred_tags(user_id)
    print(f"[INFO] user {user_id} → preferred_tags: {preferred_tags}")
    recommended_products = recommend_by_tags(preferred_tags)
    return {"user_id": user_id, "tags": preferred_tags, "products": recommended_products}

# 전체 인기 상품 반환
@app.get("/recommend/trending")
def trending(limit: int = 12):
    trending_items = get_trending_products(limit=limit)
    return {"products": trending_items}

# 연령대별 인기 상품 반환
@app.get("/recommend/age-group")
def recommend_by_age_group(range: str = "20s", gender: str = None, limit: int = 12):
    user_ids = get_users_by_age_range_and_gender(range, gender)
    print("user_ids: ", user_ids)
    products = get_popular_products_by_users(user_ids, limit)
    return {
        "range": range,
        "gender": gender,
        "user_count": len(user_ids),
        "products": products
    }

# 특정 상품 기반 유사 상품 반환
@app.get("/recommend/similar-to/{product_id}")
def recommend_similar(product_id: int, limit: int = 8):
    products = get_similar_products(product_id, limit)
    return {
        "base_product_id": product_id,
        "similar_products": products
    }

#  try-on한 착장과 유사한 상품 반환
@app.get("/recommend/tryon-based")
def recommend_tryon_based(user_id: int, limit: int = 10):
    products = get_products_similar_to_user_tryon(user_id, limit)
    return {
        "user_id": user_id,
        "products": products
    }