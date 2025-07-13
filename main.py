from fastapi import FastAPI, Query
from redis_client import get_recommendation_from_redis

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello TIO"}

# 1. 메인 페이지 개인화 추천 상품 (종합 로직)
@app.get("/api/recommend/for-you")
def recommend_for_you(user_id: int = Query(..., description="로그인한 유저의 ID"), limit: int = Query(12, description="추천 상품 개수")):
    key = f"recommend:for_you:{user_id}"
    products = get_recommendation_from_redis(key)[:limit]
    return {
        "user_id": user_id,
        "recommendation_type": "hybrid_profile_behavior",
        "products": products
    }

# 2. 메인 페이지 전체 인기 상품
@app.get("/api/recommend/trending")
def trending(limit: int = Query(12, description="추천 상품 개수")):
    key = "recommend:trending"
    products = get_recommendation_from_redis(key)[:limit]
    return {
        "recommendation_type": "trending",
        "products": products
    }

# 3. 연령대별 인기 상품
@app.get("/api/recommend/age-group")
def recommend_by_age_group(range: str = Query("20s", description="연령대 (10s, 20s, 30s, 40s)"), 
                            gender: str = Query(None, description="성별 (M, F)"), 
                            limit: int = Query(12, description="추천 상품 개수")):
    key = f"recommend:age_group:{range}:{gender or 'all'}"
    products = get_recommendation_from_redis(key)[:limit]
    return {
        "range": range,
        "gender": gender,
        "recommendation_type": "age_group_popular",
        "products": products
    }

# 4. 상세 상품 기반 유사 상품
@app.get("/api/recommend/similar-to/{product_id}")
def recommend_similar(product_id: int, limit: int = Query(8, description="추천 상품 개수")):
    key = f"recommend:similar_to:{product_id}"
    products = get_recommendation_from_redis(key)[:limit]
    return {
        "base_product_id": product_id,
        "recommendation_type": "similar_tags",
        "similar_products": products
    }

# 5. Try-on 기반 추천
@app.get("/api/recommend/tryon-based")
def recommend_tryon_based(user_id: int = Query(..., description="로그인한 유저의 ID"), 
                            limit: int = Query(10, description="추천 상품 개수")):
    key = f"recommend:tryon_based:{user_id}"
    products = get_recommendation_from_redis(key)[:limit]
    return {
        "user_id": user_id,
        "recommendation_type": "tryon_based",
        "products": products
    }