from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello TIO"}

# 1. 메인 페이지 개인화 추천 상품 (종합 로직)
@app.get("/api/recommend/for-you")
def recommend_for_you(user_id: int = Query(..., description="로그인한 유저의 ID"), limit: int = Query(12, description="추천 상품 개수")):
    """메인 페이지 개인화 추천 - 프로필 기반 + 행동 기반 하이브리드"""
    from service.collaborative_filtering_service import get_hybrid_profile_recommendations
    
    try:
        # 프로필 + 행동 기반 하이브리드 추천 (Cold Start 해결)
        products = get_hybrid_profile_recommendations(user_id, limit)
        return {
            "user_id": user_id,
            "recommendation_type": "hybrid_profile_behavior",
            "products": products
        }
    except Exception as e:
        print(f"[ERROR] 개인화 추천 실패: {e}")
        # Fallback: 인기 상품 반환
        from service.recommend_service import get_trending_products
        fallback_products = get_trending_products(limit)
        return {
            "user_id": user_id,
            "recommendation_type": "fallback_trending",
            "products": fallback_products
        }

# 2. 메인 페이지 전체 인기 상품
@app.get("/api/recommend/trending")
def trending(limit: int = Query(12, description="추천 상품 개수")):
    """전체 인기 상품 - tryon/찜/구매 수 기준"""
    from service.recommend_service import get_trending_products
    trending_items = get_trending_products(limit=limit)
    return {
        "recommendation_type": "trending",
        "products": trending_items
    }

# 3. 연령대별 인기 상품
@app.get("/api/recommend/age-group")
def recommend_by_age_group(range: str = Query("20s", description="연령대 (10s, 20s, 30s, 40s)"), 
                            gender: str = Query(None, description="성별 (M, F)"), 
                            limit: int = Query(12, description="추천 상품 개수")):
    """연령대별 인기 상품 - birthDate, gender 기준 행동 기반"""
    from service.recommend_service import get_users_by_age_range_and_gender, get_popular_products_by_users
    
    user_ids = get_users_by_age_range_and_gender(range, gender)
    print(f"[INFO] 연령대 {range}, 성별 {gender} → 사용자 수: {len(user_ids)}")
    
    products = get_popular_products_by_users(user_ids, limit)
    return {
        "range": range,
        "gender": gender,
        "user_count": len(user_ids),
        "recommendation_type": "age_group_popular",
        "products": products
    }

# 4. 상세 상품 기반 유사 상품
@app.get("/api/recommend/similar-to/{product_id}")
def recommend_similar(product_id: int, limit: int = Query(8, description="추천 상품 개수")):
    """상세 상품 기반 유사 상품 - 유사 태그 기반"""
    from service.recommend_service import get_similar_products
    
    products = get_similar_products(product_id, limit)
    return {
        "base_product_id": product_id,
        "recommendation_type": "similar_tags",
        "similar_products": products
    }

# 5. Try-on 기반 추천
@app.get("/api/recommend/tryon-based")
def recommend_tryon_based(user_id: int = Query(..., description="로그인한 유저의 ID"), 
                            limit: int = Query(10, description="추천 상품 개수")):
    """Try-on한 착장과 유사한 상품 - tryon 지연 시간에 추천 가능"""
    from service.recommend_service import get_products_similar_to_user_tryon
    
    products = get_products_similar_to_user_tryon(user_id, limit)
    return {
        "user_id": user_id,
        "recommendation_type": "tryon_based",
        "products": products
    }