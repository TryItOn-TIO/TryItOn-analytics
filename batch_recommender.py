import numpy as np
import redis
import json
import os
from typing import List, Dict, Tuple
from datetime import date
from db import get_connection

# Redis 설정
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")

redis_client = redis.StrictRedis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    db=REDIS_DB, 
    decode_responses=True
)

class BatchCollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.user_ids = []
        self.product_ids = []
        self.user_id_to_index = {}
        self.product_id_to_index = {}
        self.index_to_user_id = {}
        self.index_to_product_id = {}
        self._matrix_built = False
        self._similarity_calculated = False
        
    def _build_user_item_matrix(self):
        if self._matrix_built and self.user_item_matrix is not None:
            return True
            
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT user_id, product_id, SUM(score) as total_score
            FROM recommend_behavior_log
            GROUP BY user_id, product_id
        """)
        
        behavior_data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"[BATCH] 행동 데이터 개수: {len(behavior_data)}")
        
        if not behavior_data:
            print("[BATCH] 행동 데이터가 없습니다.")
            return False
            
        unique_users = list(set(row['user_id'] for row in behavior_data))
        unique_products = list(set(row['product_id'] for row in behavior_data))
        
        self.user_ids = sorted(unique_users)
        self.product_ids = sorted(unique_products)
        
        print(f"[BATCH] 고유 사용자 수: {len(self.user_ids)}")
        print(f"[BATCH] 고유 상품 수: {len(self.product_ids)}")
        
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.product_id_to_index = {product_id: idx for idx, product_id in enumerate(self.product_ids)}
        self.index_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
        self.index_to_product_id = {idx: product_id for product_id, idx in self.product_id_to_index.items()}
        
        self.user_item_matrix = np.zeros((len(self.user_ids), len(self.product_ids)))
        
        for row in behavior_data:
            user_idx = self.user_id_to_index[row['user_id']]
            product_idx = self.product_id_to_index[row['product_id']]
            self.user_item_matrix[user_idx, product_idx] = row['total_score']
        
        self._matrix_built = True
        return True
    
    def calculate_user_similarity(self):
        if self._similarity_calculated and self.user_similarity_matrix is not None:
            return True
            
        if self.user_item_matrix is None:
            return False
            
        # 행동 기반 유사도 계산
        user_norms = np.linalg.norm(self.user_item_matrix, axis=1, keepdims=True)
        user_norms[user_norms == 0] = 1
        normalized_matrix = self.user_item_matrix / user_norms
        behavior_similarity = np.dot(normalized_matrix, normalized_matrix.T)
        
        # 신체 정보 기반 유사도 계산
        physical_similarity = self._calculate_physical_similarity()
        
        # 두 유사도를 결합 (행동 70%, 신체 정보 30%)
        self.user_similarity_matrix = 0.7 * behavior_similarity + 0.3 * physical_similarity
        
        # 자기 자신과의 유사도는 0으로 설정
        np.fill_diagonal(self.user_similarity_matrix, 0)
        
        self._similarity_calculated = True
        return True
    
    def _calculate_physical_similarity(self):
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT member_user_id, height, weight, shoe_size, preferred_style
            FROM profile
            WHERE member_user_id IN ({})
        """.format(','.join(['%s'] * len(self.user_ids))), self.user_ids)
        
        profile_data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        profiles = {row['member_user_id']: row for row in profile_data}
        
        n_users = len(self.user_ids)
        physical_matrix = np.zeros((n_users, 4))
        
        for i, user_id in enumerate(self.user_ids):
            if user_id in profiles:
                profile = profiles[user_id]
                physical_matrix[i, 0] = profile['height'] or 0
                physical_matrix[i, 1] = profile['weight'] or 0
                physical_matrix[i, 2] = profile['shoe_size'] or 0
                style_mapping = {'CASUAL': 1, 'STREET': 2, 'HIPHOP': 3, 'CHIC': 4, 'FORMAL': 5, 'VINTAGE': 6}
                physical_matrix[i, 3] = style_mapping.get(profile['preferred_style'], 0)
        
        # 정규화
        for j in range(physical_matrix.shape[1]):
            col = physical_matrix[:, j]
            if np.std(col) > 0:
                physical_matrix[:, j] = (col - np.mean(col)) / np.std(col)
        
        # 코사인 유사도 계산
        physical_norms = np.linalg.norm(physical_matrix, axis=1, keepdims=True)
        physical_norms[physical_norms == 0] = 1
        normalized_physical = physical_matrix / physical_norms
        
        return np.dot(normalized_physical, normalized_physical.T)
    
    def get_similar_users(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if user_id not in self.user_id_to_index:
            return []
            
        user_idx = self.user_id_to_index[user_id]
        similarities = self.user_similarity_matrix[user_idx]
        
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_users = []
        for idx in similar_indices:
            if similarities[idx] > 0:
                similar_user_id = self.index_to_user_id[idx]
                similar_users.append((similar_user_id, similarities[idx]))
                
        return similar_users
    
    def get_collaborative_recommendations(self, user_id: int, top_n: int = 12) -> List[int]:
        if not self._build_user_item_matrix() or not self.calculate_user_similarity():
            return []
            
        if user_id not in self.user_id_to_index:
            return []
            
        similar_users = self.get_similar_users(user_id, top_k=20)
        
        if not similar_users:
            return []
            
        product_scores = {}
        user_idx = self.user_id_to_index[user_id]
        user_rated_products = set(np.where(self.user_item_matrix[user_idx] > 0)[0])
        
        for similar_user_id, similarity in similar_users:
            similar_user_idx = self.user_id_to_index[similar_user_id]
            rated_products = np.where(self.user_item_matrix[similar_user_idx] > 0)[0]
            
            for product_idx in rated_products:
                product_id = self.index_to_product_id[product_idx]
                base_score = self.user_item_matrix[similar_user_idx, product_idx] * similarity
                
                if product_idx in user_rated_products:
                    score = base_score * 0.1
                else:
                    score = base_score
                
                if product_id in product_scores:
                    product_scores[product_id] += score
                else:
                    product_scores[product_id] = score
        
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        return [product_id for product_id, _ in sorted_products[:top_n]]

class BatchProfileBasedRecommender:
    def __init__(self):
        self.profile_matrix = None
        self.profile_similarity_matrix = None
        self.user_ids = []
        self.user_id_to_index = {}
        self.index_to_user_id = {}
        self._matrix_built = False
        self._similarity_calculated = False
    
    def _build_profile_matrix(self):
        if self._matrix_built and self.profile_matrix is not None:
            return True
            
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT member_user_id, height, weight, shoe_size, preferred_style
            FROM profile
        """)
        
        profile_data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not profile_data:
            return False
            
        self.user_ids = sorted([row['member_user_id'] for row in profile_data])
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.index_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
        
        n_users = len(self.user_ids)
        self.profile_matrix = np.zeros((n_users, 4))
        
        profiles = {row['member_user_id']: row for row in profile_data}
        
        for i, user_id in enumerate(self.user_ids):
            if user_id in profiles:
                profile = profiles[user_id]
                self.profile_matrix[i, 0] = profile['height'] or 0
                self.profile_matrix[i, 1] = profile['weight'] or 0
                self.profile_matrix[i, 2] = profile['shoe_size'] or 0
                style_mapping = {'CASUAL': 1, 'STREET': 2, 'HIPHOP': 3, 'CHIC': 4, 'FORMAL': 5, 'VINTAGE': 6}
                self.profile_matrix[i, 3] = style_mapping.get(profile['preferred_style'], 0)
        
        # 정규화
        for j in range(self.profile_matrix.shape[1]):
            col = self.profile_matrix[:, j]
            if np.std(col) > 0:
                self.profile_matrix[:, j] = (col - np.mean(col)) / np.std(col)
        
        self._matrix_built = True
        return True
    
    def calculate_profile_similarity(self):
        if self._similarity_calculated and self.profile_similarity_matrix is not None:
            return True
            
        if not self._build_profile_matrix():
            return False
            
        # 코사인 유사도 계산
        norms = np.linalg.norm(self.profile_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_matrix = self.profile_matrix / norms
        
        self.profile_similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
        np.fill_diagonal(self.profile_similarity_matrix, 0)
        
        self._similarity_calculated = True
        return True
    
    def get_similar_users_by_profile(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if user_id not in self.user_id_to_index:
            return []
            
        user_idx = self.user_id_to_index[user_id]
        similarities = self.profile_similarity_matrix[user_idx]
        
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_users = []
        for idx in similar_indices:
            if similarities[idx] > 0:
                similar_user_id = self.index_to_user_id[idx]
                similar_users.append((similar_user_id, similarities[idx]))
                
        return similar_users
    
    def get_profile_based_recommendations(self, user_id: int, top_n: int = 12) -> List[int]:
        if not self.calculate_profile_similarity():
            return []
            
        similar_users = self.get_similar_users_by_profile(user_id, top_k=15)
        
        if not similar_users:
            return []
            
        # 유사한 사용자들이 선호하는 상품들 수집
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        similar_user_ids = [user_id for user_id, _ in similar_users]
        format_strings = ','.join(['%s'] * len(similar_user_ids))
        
        cursor.execute(f"""
            SELECT product_id, SUM(score) as total_score
            FROM recommend_behavior_log
            WHERE user_id IN ({format_strings})
            GROUP BY product_id
            ORDER BY total_score DESC
            LIMIT %s
        """, similar_user_ids + [top_n * 2])
        
        product_scores = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [row['product_id'] for row in product_scores[:top_n]]

class BatchRecommendationService:
    def __init__(self):
        self.cf_recommender = BatchCollaborativeFilteringRecommender()
        self.profile_recommender = BatchProfileBasedRecommender()
    
    def get_trending_products(self, limit=12):
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
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
    
    def get_users_by_age_range_and_gender(self, age_range: str, gender: str = None) -> list:
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
    
    def get_popular_products_by_users(self, user_ids: list, limit: int = 12):
        if not user_ids:
            return []
        
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        format_str = ','.join(['%s'] * len(user_ids))
        
        query = f"""
        SELECT product_id, SUM(score) AS total_score
        FROM (
            SELECT wi.product_id, COUNT(*) * 1 AS score
            FROM wishlist_item wi
            JOIN wishlist w ON wi.wishlist_id = w.wishlist_id
            WHERE w.user_id IN ({format_str})
            GROUP BY wi.product_id
            
            UNION ALL
            
            SELECT oi.product_id, COUNT(*) * 3 AS score
            FROM order_item oi
            JOIN orders o ON oi.order_id = o.order_id
            WHERE o.user_id IN ({format_str})
            GROUP BY oi.product_id
            
            UNION ALL
            
            SELECT ai.product_id, COUNT(*) * 2 AS score
            FROM avatar_item ai
            JOIN avatar a ON ai.avatar_id = a.avatar_id
            WHERE a.user_id IN ({format_str})
            GROUP BY ai.product_id
            
            UNION ALL
            
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
    
    def get_similar_products(self, product_id: int, limit: int = 8):
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT tag_id FROM product_tag WHERE product_id = %s
        """, (product_id,))
        base_tag_ids = [row["tag_id"] for row in cursor.fetchall()]
        
        if not base_tag_ids:
            return []
        
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
    
    def get_products_similar_to_user_tryon(self, user_id: int, limit: int = 10):
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT DISTINCT ai.product_id
            FROM avatar_item ai
            JOIN avatar a ON ai.avatar_id = a.avatar_id
            WHERE a.user_id = %s
        """, (user_id,))
        tried_products = [row["product_id"] for row in cursor.fetchall()]
        
        if not tried_products:
            return []
        
        format_ids = ','.join(['%s'] * len(tried_products))
        cursor.execute(f"""
            SELECT DISTINCT tag_id
            FROM product_tag
            WHERE product_id IN ({format_ids})
        """, tried_products)
        tag_ids = [row["tag_id"] for row in cursor.fetchall()]
        
        if not tag_ids:
            return []
        
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
        
        products = cursor.fetchall()
        cursor.close()
        conn.close()
        return products
    
    def get_hybrid_recommendations(self, user_id: int, top_n: int = 12) -> List[Dict]:
        # 협업필터링 추천 (60%)
        cf_recommendations = self.cf_recommender.get_collaborative_recommendations(user_id, top_n)
        
        # 프로필 기반 추천 (40%)
        profile_recommendations = self.profile_recommender.get_profile_based_recommendations(user_id, top_n)
        
        # 결과 합치기 (중복 제거)
        all_recommendations = []
        seen_products = set()
        
        # 협업필터링 결과 먼저 추가
        for product_id in cf_recommendations:
            if product_id not in seen_products:
                all_recommendations.append(product_id)
                seen_products.add(product_id)
        
        # 프로필 기반 결과 추가
        for product_id in profile_recommendations:
            if product_id not in seen_products and len(all_recommendations) < top_n:
                all_recommendations.append(product_id)
                seen_products.add(product_id)
        
        # 상품 상세 정보 조회
        if all_recommendations:
            conn = get_connection()
            cursor = conn.cursor(dictionary=True)
            
            format_strings = ','.join(['%s'] * len(all_recommendations))
            cursor.execute(f"""
                SELECT product_id, product_name, brand, img1
                FROM product
                WHERE product_id IN ({format_strings})
            """, all_recommendations)
            
            products = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return products
        
        return []

def store_recommendations_in_redis():
    """모든 추천 결과를 계산하고 Redis에 저장하는 메인 함수"""
    print("[BATCH] 배치 추천 계산 시작...")
    
    service = BatchRecommendationService()
    
    # 1. 트렌딩 상품 계산 및 저장
    print("[BATCH] 트렌딩 상품 계산 중...")
    trending_products = service.get_trending_products(50)
    redis_client.set("recommend:trending", json.dumps(trending_products))
    print(f"[BATCH] 트렌딩 상품 {len(trending_products)}개 저장 완료")
    
    # 2. 연령대별 인기 상품 계산 및 저장
    print("[BATCH] 연령대별 인기 상품 계산 중...")
    age_ranges = ["10s", "20s", "30s", "40s"]
    genders = ["M", "F", None]
    
    for age_range in age_ranges:
        for gender in genders:
            user_ids = service.get_users_by_age_range_and_gender(age_range, gender)
            if user_ids:
                products = service.get_popular_products_by_users(user_ids, 50)
                gender_key = gender or "all"
                key = f"recommend:age_group:{age_range}:{gender_key}"
                redis_client.set(key, json.dumps(products))
                print(f"[BATCH] 연령대별 상품 {age_range}:{gender_key} - {len(products)}개 저장 완료")
    
    # 3. 모든 상품에 대한 유사 상품 계산 및 저장
    print("[BATCH] 유사 상품 계산 중...")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT product_id FROM product LIMIT 1000")  # 상위 1000개 상품만
    product_ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    for product_id in product_ids:
        similar_products = service.get_similar_products(product_id, 20)
        key = f"recommend:similar_to:{product_id}"
        redis_client.set(key, json.dumps(similar_products))
    
    print(f"[BATCH] 유사 상품 {len(product_ids)}개 상품에 대해 계산 완료")
    
    # 4. 개인화 추천 계산 및 저장 (활성 사용자들에 대해)
    print("[BATCH] 개인화 추천 계산 중...")
    
    # 활성 사용자 목록 조회
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT user_id 
        FROM recommend_behavior_log 
        GROUP BY user_id 
        HAVING COUNT(*) >= 3
        LIMIT 1000
    """)
    active_user_ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    print(f"[BATCH] 활성 사용자 {len(active_user_ids)}명에 대해 개인화 추천 계산")
    
    for user_id in active_user_ids:
        # 하이브리드 추천 (프로필 + 행동)
        hybrid_products = service.get_hybrid_recommendations(user_id, 20)
        key = f"recommend:for_you:{user_id}"
        redis_client.set(key, json.dumps(hybrid_products))
        
        # Try-on 기반 추천
        tryon_products = service.get_products_similar_to_user_tryon(user_id, 20)
        key = f"recommend:tryon_based:{user_id}"
        redis_client.set(key, json.dumps(tryon_products))
    
    print(f"[BATCH] 개인화 추천 {len(active_user_ids)}명 사용자에 대해 계산 완료")
    
    # 5. 캐시 만료 시간 설정 (24시간)
    print("[BATCH] 캐시 만료 시간 설정 중...")
    for key in redis_client.keys("recommend:*"):
        redis_client.expire(key, 86400)  # 24시간
    
    print("[BATCH] 배치 추천 계산 완료!")

if __name__ == "__main__":
    store_recommendations_in_redis() 