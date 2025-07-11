from collections import defaultdict
from db import get_connection
from user_profile import get_user_preferred_tags
from user_behavior import get_user_behavior_logs
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 상품 태그 기반 Content 추천을 위한 벡터화 및 추천 기능을 제공하는 클래스
class VectorRecommender:
    # 인스턴스 생성 시 전체 상품 태그 매트릭스와 매핑 정보 초기화
    def __init__(self):
        self.product_matrix, self.product_ids, self.mlb = self._build_product_tag_matrix()

    # DB에서 전체 상품의 태그 정보를 조회하여 dict 형태로 반환
    @staticmethod
    def load_product_tags():
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT pt.product_id, t.tag_name
            FROM product_tag pt
            JOIN tag t ON pt.tag_id = t.tag_id
            """
        )
        tag_map = defaultdict(list)
        for row in cursor.fetchall():
            tag_map[row['product_id']].append(row['tag_name'])
        cursor.close()
        conn.close()
        # print("DEBUG load_product_tags -> tag_map:", dict(tag_map))
        return tag_map

    # 전체 상품 태그를 로드하고 MultiLabelBinarizer로 벡터화 -> multi-hot 벡터
    def _build_product_tag_matrix(self):
        tag_map = self.load_product_tags()
        product_ids = list(tag_map.keys())
        tag_lists = [tag_map[pid] for pid in product_ids]

        mlb = MultiLabelBinarizer()
        matrix = mlb.fit_transform(tag_lists)
        # print("DEBUG _build_product_tag_matrix -> product_ids:", product_ids)
        # print("DEBUG _build_product_tag_matrix -> mlb.classes_:", mlb.classes_)
        # print("DEBUG _build_product_tag_matrix -> matrix shape:", matrix.shape)
        return matrix, product_ids, mlb

    # 유저 프로필 태그와 행동 로그를 기반으로 태그별 가중치를 반영한 벡터 반환
    def _build_user_vector(self, user_id: int):
        # 프로필 기반 선호 태그
        pref_tags = get_user_preferred_tags(user_id)

        # 행동 로그 기반
        behavior_logs = get_user_behavior_logs(user_id)

        # 전체 태그 차원
        D = len(self.mlb.classes_)
        user_vec = np.zeros((1, D), dtype=float)
        
        # 프로필 태그 가중치 추가
        for tag in pref_tags:
            if tag in self.mlb.classes_:
                idx = list(self.mlb.classes_).index(tag)
                user_vec[0, idx] += 1.0

        # 행동 로그 점수 추가
        for tag, score in behavior_logs:
            if tag in self.mlb.classes_:
                idx = list(self.mlb.classes_).index(tag)
                user_vec[0, idx] += score

        print("DEBUG weighted user_vec:", user_vec)
        return user_vec

    # user_id에 대해 추천 상품 ID 리스트 반환
    def recommend_by_user(self, user_id: int, top_n: int = 8):
        print(f"DEBUG recommend_by_user -> user_id: {user_id}, top_n: {top_n}")
        user_vec = self._build_user_vector(user_id)
        print("DEBUG recommend_by_user -> user_vec:", user_vec)
        if user_vec is None:
            return []
        sims = cosine_similarity(user_vec, self.product_matrix).flatten()
        print("DEBUG recommend_by_user -> sims:", sims)
        top_indices = sims.argsort()[::-1][:top_n]
        print("DEBUG recommend_by_user -> top_indices:", top_indices)
        return [self.product_ids[i] for i in top_indices]