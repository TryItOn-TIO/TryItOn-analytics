import numpy as np
from typing import List, Dict, Tuple
from db import get_connection
from repository import get_product_details, get_category_map
from service.recommend_service import get_trending_products, recommend_by_tags
from user_profile import get_user_preferred_tags

class CollaborativeFilteringRecommender:
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
        
    # 사용자-상품 행렬을 구축합니다.
    def _build_user_item_matrix(self):
        # 이미 구축된 경우 캐시된 결과 반환
        if self._matrix_built and self.user_item_matrix is not None:
            return True
            
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 모든 사용자와 상품의 행동 데이터를 가져옵니다
        cursor.execute("""
            SELECT user_id, product_id, SUM(score) as total_score
            FROM recommend_behavior_log
            GROUP BY user_id, product_id
        """)
        
        behavior_data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"[DEBUG] 행동 데이터 개수: {len(behavior_data)}")
        if behavior_data:
            print(f"[DEBUG] 첫 번째 데이터: {behavior_data[0]}")
        
        if not behavior_data:
            print("[DEBUG] 행동 데이터가 없습니다.")
            return False
            
        # 사용자 ID와 상품 ID의 고유한 리스트를 생성
        unique_users = list(set(row['user_id'] for row in behavior_data))
        unique_products = list(set(row['product_id'] for row in behavior_data))
        
        self.user_ids = sorted(unique_users)
        self.product_ids = sorted(unique_products)
        
        print(f"[DEBUG] 고유 사용자 수: {len(self.user_ids)}, 사용자 ID: {self.user_ids}")
        print(f"[DEBUG] 고유 상품 수: {len(self.product_ids)}")
        
        # ID를 인덱스로 매핑
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.product_id_to_index = {product_id: idx for idx, product_id in enumerate(self.product_ids)}
        self.index_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
        self.index_to_product_id = {idx: product_id for product_id, idx in self.product_id_to_index.items()}
        
        # 사용자-상품 행렬 초기화
        self.user_item_matrix = np.zeros((len(self.user_ids), len(self.product_ids)))
        
        # 행렬에 데이터 채우기
        for row in behavior_data:
            user_idx = self.user_id_to_index[row['user_id']]
            product_idx = self.product_id_to_index[row['product_id']]
            self.user_item_matrix[user_idx, product_idx] = row['total_score']
        
        self._matrix_built = True
        return True
    
    # 사용자 간 유사도를 계산합니다 (행동 + 신체 정보 기반)
    def calculate_user_similarity(self):
        # 이미 계산된 경우 캐시된 결과 반환
        if self._similarity_calculated and self.user_similarity_matrix is not None:
            return True
            
        if self.user_item_matrix is None:
            return False
            
        # 1. 행동 기반 유사도 계산
        user_norms = np.linalg.norm(self.user_item_matrix, axis=1, keepdims=True)
        user_norms[user_norms == 0] = 1  # 0으로 나누기 방지
        normalized_matrix = self.user_item_matrix / user_norms
        behavior_similarity = np.dot(normalized_matrix, normalized_matrix.T)
        
        # 2. 신체 정보 기반 유사도 계산
        physical_similarity = self._calculate_physical_similarity()
        
        print(f"[DEBUG] 행동 기반 유사도 범위: {np.min(behavior_similarity):.3f} ~ {np.max(behavior_similarity):.3f}")
        print(f"[DEBUG] 신체 정보 기반 유사도 범위: {np.min(physical_similarity):.3f} ~ {np.max(physical_similarity):.3f}")
        
        # 3. 두 유사도를 결합 (행동 70%, 신체 정보 30%)
        self.user_similarity_matrix = 0.7 * behavior_similarity + 0.3 * physical_similarity
        
        print(f"[DEBUG] 결합된 유사도 범위: {np.min(self.user_similarity_matrix):.3f} ~ {np.max(self.user_similarity_matrix):.3f}")
        
        # 자기 자신과의 유사도는 0으로 설정
        np.fill_diagonal(self.user_similarity_matrix, 0)
        
        self._similarity_calculated = True
        return True
    
    # 신체 정보 기반 사용자 유사도를 계산합니다
    def _calculate_physical_similarity(self):
        # 프로필 데이터 가져오기
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
        
        # 프로필 데이터를 딕셔너리로 변환
        profiles = {row['member_user_id']: row for row in profile_data}
        
        # 신체 정보 행렬 생성
        n_users = len(self.user_ids)
        physical_matrix = np.zeros((n_users, 4))  # height, weight, shoe_size, style
        
        for i, user_id in enumerate(self.user_ids):
            if user_id in profiles:
                profile = profiles[user_id]
                physical_matrix[i, 0] = profile['height'] or 0
                physical_matrix[i, 1] = profile['weight'] or 0
                physical_matrix[i, 2] = profile['shoe_size'] or 0
                # 스타일은 원핫 인코딩 대신 간단히 숫자로 매핑
                style_mapping = {'CASUAL': 1, 'STREET': 2, 'HIPHOP': 3, 'CHIC': 4, 'FORMAL': 5, 'VINTAGE': 6}
                physical_matrix[i, 3] = style_mapping.get(profile['preferred_style'], 0)
        
        # 정규화 (각 특성별로 정규화)
        for j in range(physical_matrix.shape[1]):
            col = physical_matrix[:, j]
            if np.std(col) > 0:
                physical_matrix[:, j] = (col - np.mean(col)) / np.std(col)
        
        # 코사인 유사도 계산
        physical_norms = np.linalg.norm(physical_matrix, axis=1, keepdims=True)
        physical_norms[physical_norms == 0] = 1
        normalized_physical = physical_matrix / physical_norms
        
        return np.dot(normalized_physical, normalized_physical.T)
    
    # 캐시를 초기화합니다. 새로운 데이터가 추가되었을 때 호출합니다.
    def clear_cache(self):
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self._matrix_built = False
        self._similarity_calculated = False
    
    # 주어진 사용자와 유사한 사용자들을 반환합니다.
    def get_similar_users(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if user_id not in self.user_id_to_index:
            return []
            
        user_idx = self.user_id_to_index[user_id]
        similarities = self.user_similarity_matrix[user_idx]
        
        # 유사도가 높은 순으로 정렬
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_users = []
        for idx in similar_indices:
            if similarities[idx] > 0:  # 유사도가 0보다 큰 경우만
                similar_user_id = self.index_to_user_id[idx]
                similar_users.append((similar_user_id, similarities[idx]))
                
        return similar_users
    
    # 협업필터링을 통한 추천 상품을 반환합니다.
    def get_collaborative_recommendations(self, user_id: int, top_n: int = 12, diversity_factor: float = 0.3) -> List[int]:
        print(f"[DEBUG] 협업필터링 시작 - user_id: {user_id}")
        
        if not self._build_user_item_matrix() or not self.calculate_user_similarity():
            print("[DEBUG] 행렬 구축 또는 유사도 계산 실패")
            return []
            
        if user_id not in self.user_id_to_index:
            print(f"[DEBUG] 사용자 {user_id}가 행렬에 없습니다. 사용 가능한 사용자: {self.user_ids}")
            return []
            
        # 유사한 사용자들 찾기
        similar_users = self.get_similar_users(user_id, top_k=20)
        print(f"[DEBUG] 유사한 사용자 수: {len(similar_users)}")
        
        if not similar_users:
            print("[DEBUG] 유사한 사용자가 없습니다.")
            return []
            
        # 유사한 사용자들이 선호하는 상품들의 점수 계산
        product_scores = {}
        user_idx = self.user_id_to_index[user_id]
        user_rated_products = set(np.where(self.user_item_matrix[user_idx] > 0)[0])
        
        print(f"[DEBUG] 사용자 {user_id}가 평가한 상품 수: {len(user_rated_products)}")
        print(f"[DEBUG] 사용자 {user_id}가 평가한 상품 인덱스: {list(user_rated_products)}")
        
        for similar_user_id, similarity in similar_users:
            similar_user_idx = self.user_id_to_index[similar_user_id]
            
            # 해당 사용자가 평가한 상품들
            rated_products = np.where(self.user_item_matrix[similar_user_idx] > 0)[0]
            print(f"[DEBUG] 유사 사용자 {similar_user_id} (유사도: {similarity:.3f})가 평가한 상품 수: {len(rated_products)}")
            
            for product_idx in rated_products:
                product_id = self.index_to_product_id[product_idx]
                base_score = self.user_item_matrix[similar_user_idx, product_idx] * similarity
                
                # 사용자가 이미 평가한 상품은 낮은 가중치 적용
                if product_idx in user_rated_products:
                    # 이미 평가한 상품은 0.1 가중치 적용
                    score = base_score * 0.1
                else:
                    # 새로운 상품은 원래 점수 적용
                    score = base_score
                
                if product_id in product_scores:
                    product_scores[product_id] += score
                else:
                    product_scores[product_id] = score
        
        print(f"[DEBUG] 추천 후보 상품 수: {len(product_scores)}")
        if product_scores:
            print(f"[DEBUG] 추천 후보 상품 점수 (상위 5개): {sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        # 점수 순으로 정렬
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 다양성을 위한 카테고리별 제한
        if diversity_factor > 0:
            return self._diversify_recommendations(sorted_products, top_n, diversity_factor)
        else:
            return [product_id for product_id, _ in sorted_products[:top_n]]
    
    # 카테고리 다양성을 고려하여 추천을 다변화합니다.
    def _diversify_recommendations(self, sorted_products: List[Tuple[int, float]], top_n: int, diversity_factor: float) -> List[int]:
        if not sorted_products:
            return []
            
        # 상품들의 카테고리 정보 가져오기
        product_ids = [product_id for product_id, _ in sorted_products]
        category_map = get_category_map(product_ids)
        
        # 카테고리별로 상품들을 그룹화
        category_products = {}
        for product_id, score in sorted_products:
            category = category_map.get(product_id, "unknown")
            if category not in category_products:
                category_products[category] = []
            category_products[category].append((product_id, score))
        
        # 카테고리별 최대 상품 수 계산
        max_per_category = max(1, int(top_n * diversity_factor))
        
        # 각 카테고리에서 상품을 선택
        selected_products = []
        for category, products in category_products.items():
            category_selection = products[:max_per_category]
            selected_products.extend(category_selection)
        
        # 최종 점수 순으로 정렬하고 top_n개 선택
        selected_products.sort(key=lambda x: x[1], reverse=True)
        return [product_id for product_id, _ in selected_products[:top_n]]
    
    # 사용자가 아직 접해보지 않은 카테고리의 상품을 우선적으로 추천합니다.
    def get_category_diverse_recommendations(self, user_id: int, top_n: int = 12) -> List[int]:
        if not self._build_user_item_matrix() or not self.calculate_user_similarity():
            return []
            
        if user_id not in self.user_id_to_index:
            return []
            
        # 사용자가 이미 상호작용한 상품들의 카테고리 파악
        user_idx = self.user_id_to_index[user_id]
        user_rated_products = np.where(self.user_item_matrix[user_idx] > 0)[0]
        user_product_ids = [self.index_to_product_id[idx] for idx in user_rated_products]
        
        user_category_map = get_category_map(user_product_ids)
        user_categories = set(user_category_map.values())
        
        # 유사한 사용자들이 선호하는 상품들 중 사용자가 접해보지 않은 카테고리 우선 추천
        similar_users = self.get_similar_users(user_id, top_k=15)
        
        if not similar_users:
            return []
            
        # 카테고리별 점수 계산
        category_scores = {}
        product_scores = {}
        
        for similar_user_id, similarity in similar_users:
            similar_user_idx = self.user_id_to_index[similar_user_id]
            rated_products = np.where(self.user_item_matrix[similar_user_idx] > 0)[0]
            
            for product_idx in rated_products:
                if product_idx in user_rated_products:
                    continue
                    
                product_id = self.index_to_product_id[product_idx]
                score = self.user_item_matrix[similar_user_idx, product_idx] * similarity
                
                # 카테고리 정보 가져오기
                category = get_category_map([product_id]).get(product_id, "unknown")
                
                # 새로운 카테고리에 가중치 부여
                diversity_bonus = 1.5 if category not in user_categories else 1.0
                adjusted_score = score * diversity_bonus
                
                if product_id in product_scores:
                    product_scores[product_id] += adjusted_score
                else:
                    product_scores[product_id] = adjusted_score
                
                if category in category_scores:
                    category_scores[category] += adjusted_score
                else:
                    category_scores[category] = adjusted_score
        
        # 점수 순으로 정렬
        sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [product_id for product_id, _ in sorted_products[:top_n]]


# Profile 기반 유사도 추천을 위한 새로운 클래스
class ProfileBasedRecommender:
    def __init__(self):
        self.profile_matrix = None
        self.user_ids = []
        self.user_id_to_index = {}
        self.index_to_user_id = {}
        self._matrix_built = False
        
    # 모든 사용자의 프로필 정보를 행렬로 구축합니다.
    def _build_profile_matrix(self):
        if self._matrix_built and self.profile_matrix is not None:
            return True
            
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 모든 사용자의 프로필 정보를 가져옵니다
        cursor.execute("""
            SELECT member_user_id, height, weight, shoe_size, preferred_style
            FROM profile
            WHERE height IS NOT NULL AND weight IS NOT NULL
        """)
        
        profile_data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not profile_data:
            return False
            
        # 사용자 ID 리스트 생성
        self.user_ids = sorted([row['member_user_id'] for row in profile_data])
        
        # ID를 인덱스로 매핑
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.index_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_index.items()}
        
        # 프로필 행렬 생성 (height, weight, shoe_size, style)
        n_users = len(self.user_ids)
        self.profile_matrix = np.zeros((n_users, 4))
        
        # 스타일 매핑
        style_mapping = {'CASUAL': 1, 'STREET': 2, 'HIPHOP': 3, 'CHIC': 4, 'FORMAL': 5, 'VINTAGE': 6}
        
        for row in profile_data:
            user_idx = self.user_id_to_index[row['member_user_id']]
            self.profile_matrix[user_idx, 0] = row['height'] or 0
            self.profile_matrix[user_idx, 1] = row['weight'] or 0
            self.profile_matrix[user_idx, 2] = row['shoe_size'] or 0
            self.profile_matrix[user_idx, 3] = style_mapping.get(row['preferred_style'], 0)
        
        # 정규화 (각 특성별로 정규화)
        for j in range(self.profile_matrix.shape[1]):
            col = self.profile_matrix[:, j]
            if np.std(col) > 0:
                self.profile_matrix[:, j] = (col - np.mean(col)) / np.std(col)
        
        self._matrix_built = True
        return True
    
    # 프로필 기반으로 유사한 사용자들을 찾습니다.
    def get_similar_users_by_profile(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self._build_profile_matrix():
            return []
            
        if user_id not in self.user_id_to_index:
            return []
            
        user_idx = self.user_id_to_index[user_id]
        user_profile = self.profile_matrix[user_idx].reshape(1, -1)
        
        # 모든 사용자와의 코사인 유사도 계산
        similarities = []
        for i, other_user_id in enumerate(self.user_ids):
            if i != user_idx:  # 자기 자신 제외
                other_profile = self.profile_matrix[i].reshape(1, -1)
                similarity = np.dot(user_profile, other_profile.T)[0, 0]
                similarities.append((other_user_id, similarity))
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    # 프로필 기반 추천 상품을 반환합니다.
    def get_profile_based_recommendations(self, user_id: int, top_n: int = 12) -> List[int]:
        # 유사한 사용자들 찾기
        similar_users = self.get_similar_users_by_profile(user_id, top_k=20)
        
        if not similar_users:
            return []
        
        # 유사한 사용자들이 구매한 상품들 수집
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        similar_user_ids = [user_id for user_id, _ in similar_users]
        placeholder = ','.join(['%s'] * len(similar_user_ids))
        
        cursor.execute(f"""
            SELECT product_id, COUNT(*) as purchase_count
            FROM recommend_behavior_log
            WHERE user_id IN ({placeholder})
            GROUP BY product_id
            ORDER BY purchase_count DESC
        """, similar_user_ids)
        
        product_counts = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # 상품 ID 리스트 반환
        return [row['product_id'] for row in product_counts[:top_n]]
    
    # 캐시를 초기화합니다.
    def clear_cache(self):
        self.profile_matrix = None
        self._matrix_built = False


# 전역 인스턴스
collaborative_recommender = CollaborativeFilteringRecommender()
profile_recommender = ProfileBasedRecommender()


# 협업필터링 기반 추천 상품을 반환합니다.
def get_collaborative_recommendations(user_id: int, top_n: int = 12) -> List[Dict]:
    recommended_product_ids = collaborative_recommender.get_collaborative_recommendations(
        user_id, top_n
    )
    print("get_collaborative_recommendations: ", recommended_product_ids)
    
    if not recommended_product_ids:
        print("[INFO] 협업필터링 추천이 없어서 인기 상품으로 대체합니다.")
        # 협업필터링이 실패하면 인기 상품 추천
        trending_products = get_trending_products(top_n)
        return trending_products
    
    return get_product_details(recommended_product_ids)


# 카테고리 다양성을 고려한 협업필터링 추천을 반환합니다.
def get_category_diverse_recommendations(user_id: int, top_n: int = 12) -> List[Dict]:
    recommended_product_ids = collaborative_recommender.get_category_diverse_recommendations(
        user_id, top_n
    )
    print("get_category_diverse_recommendations: ", recommended_product_ids)
    
    if not recommended_product_ids:
        return []
    
    return get_product_details(recommended_product_ids)


# 협업필터링과 기존 추천을 결합한 하이브리드 추천을 반환합니다.
def get_hybrid_collaborative_recommendations(user_id: int, top_n: int = 12) -> List[Dict]:
    # 협업필터링 추천 (60%)
    collaborative_count = int(top_n * 0.6)
    collaborative_products = get_collaborative_recommendations(user_id, collaborative_count)
    print("협업필터링 추천 (60%): ", collaborative_products)
    
    # 기존 태그 기반 추천 (40%)
    content_count = top_n - collaborative_count
    preferred_tags = get_user_preferred_tags(user_id)
    content_products = recommend_by_tags(preferred_tags)[:content_count]
    print("기존 태그 기반 추천 (40%): ", content_products)
    
    # 결과 합치기
    hybrid_products = collaborative_products + content_products
    print("전체: ", hybrid_products)
    
    return hybrid_products[:top_n] 


# 프로필 기반 추천 상품을 반환합니다.
def get_profile_based_recommendations(user_id: int, top_n: int = 12) -> List[Dict]:
    from repository import get_product_details
    
    recommended_product_ids = profile_recommender.get_profile_based_recommendations(
        user_id, top_n
    )
    print(f"[DEBUG] 프로필 기반 추천 상품 ID: {recommended_product_ids}")
    
    if not recommended_product_ids:
        return []
    
    return get_product_details(recommended_product_ids)


def get_hybrid_profile_recommendations(user_id: int, top_n: int = 12) -> List[Dict]:
    """프로필 기반과 행동 기반을 결합한 하이브리드 추천을 반환합니다."""
    from service.recommend_service import get_trending_products
    
    # 프로필 기반 추천 (70%)
    profile_count = int(top_n * 0.7)
    profile_products = get_profile_based_recommendations(user_id, profile_count)
    
    # 행동 기반 협업필터링 추천 (30%)
    behavior_count = top_n - profile_count
    behavior_products = get_collaborative_recommendations(user_id, behavior_count)
    
    # 결과 합치기
    hybrid_products = profile_products + behavior_products
    print(f"[DEBUG] 하이브리드 추천 - 프로필: {len(profile_products)}, 행동: {len(behavior_products)}")
    
    return hybrid_products[:top_n] 