from collections import defaultdict
from db import get_connection
from user_profile import get_user_preferred_tags
from user_behavior import get_user_behavior_logs
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# 상품 태그 기반 Content 추천을 위한 벡터화 및 추천 기능을 제공하는 클래스
class VectorRecommender:
    # 인스턴스 생성 시 전체 상품 태그 매트릭스와 매핑 정보 초기화
    def __init__(self):
        self.product_matrix, self.product_ids, self.mlb = self._build_product_tag_matrix()

    # DB에서 전체 상품의 카테고리와 태그 정보를 조회하여 dict 형태로 반환
    @staticmethod
    def load_product_tags():
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        # 상품 태그 조회
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
        
        # 상품 카테고리 조회 및 태그에 추가
        cursor.execute(
            """
            SELECT p.product_id, c.category_name
            FROM product p
            JOIN category c ON p.category_id = c.category_id
            """
        )
        for row in cursor.fetchall():
            pid = row['product_id']
            if pid in tag_map:
                tag_map[pid].append(row['category_name'])

        cursor.close()
        conn.close()
        # print("load_product_tags -> tag_map:", dict(tag_map))
        keys = list(tag_map.keys())
        values = list(tag_map.values())
        for i in range(10):
            print(f"1. DB 상품 정보에서 태그 정보 가져오기 \n key : {keys[i]}", values[i])
        return tag_map

    # 전체 상품 태그를 로드하고 MultiLabelBinarizer로 벡터화 -> multi-hot 벡터
    def _build_product_tag_matrix(self):
        tag_map = self.load_product_tags()
        print("**********************************************")
        product_ids = list(tag_map.keys())
        tag_lists = [tag_map[pid] for pid in product_ids]

        mlb = MultiLabelBinarizer()
        matrix = mlb.fit_transform(tag_lists)
        print("로드한 상품의 id(10개)\n ", product_ids[0:10])
        print("**********************************************")
        print("MultiLabelBinarizer가 fit 과정에서 학습한 전체 고유 태그(클래스)의 리스트\n", mlb.classes_)
        print("**********************************************")
        print("rows : 샘플 수(태그 리스트를 벡터화한 상품 개수), cols : 특성 수(고유 태그의 수)\n", matrix.shape)
        df = pd.DataFrame(
            matrix,
            index=product_ids,
            columns=mlb.classes_
        )
        print("matrix\n", df.head(10))
        print("**********************************************")
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

        print("가중치 포함한 유저 벡터\n", user_vec)
        print("**********************************************")
        return user_vec

    # user_id에 대해 추천 상품 ID 리스트 반환
    def recommend_by_user(self, user_id: int, top_n: int = 8):
        print(f"user_id: {user_id}, 상위 {top_n}개")
        user_vec = self._build_user_vector(user_id)
        print("user_vec:", user_vec)
        if user_vec is None:
            return []
        sims = cosine_similarity(user_vec, self.product_matrix).flatten()
        print("**********************************************")
        print("cosine_similarity 결과\n", sims)
        top_indices = sims.argsort()[::-1][:top_n]
        print("**********************************************")
        recommended_ids = [self.product_ids[i] for i in top_indices]
        print(f"상위 {top_n}개\n", recommended_ids)
        return [self.product_ids[i] for i in top_indices]