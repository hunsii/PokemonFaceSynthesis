import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import numpy as np

def get_facearea(image):
    p = [
        # 얼굴 6각형(시계), 눈 위(좌우), 눈 아래(좌우), 코 중앙, 입술(시계)
        10, 368, 397, 152, 172, 162, 
        # 223, 442, 24, 253, 
        # 4, 
        # 0, 287, 17, 57
    ]
    pts1 = []
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    face_detector = vision.FaceLandmarker.create_from_options(options)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape

    # STEP 3: Load the input image.
    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # STEP 4: Detect face landmarks from the input image.
    detection_result = face_detector.detect(image_mp)
    face_landmarks_list = detection_result.face_landmarks
    face_blendshapes_list = detection_result.face_blendshapes

    # Loop through the detected faces to visualize.
    for face_landmarks in face_landmarks_list:
        for idx, landmark in enumerate(face_landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * w)
            if idx in p:
                pts1.append((x, y))
            # print(image.shape)
            # image = cv2.circle(image, (x, y), 3, (255, 255, 255), 3)

    
    def draw_landmarks_on_image(rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())

        return annotated_image

    mask_rgb = np.zeros_like(image, dtype=np.uint8)
    mask_rgb = draw_landmarks_on_image(mask_rgb, detection_result)

    mask = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    for i in range(512):
        start_idx = 0
        for j in range(512):
            # print(mask[i, j])
            if mask[i, j] > 127:
                start_idx = j
                break
        end_idx = -1
        for j in range(511, -1, -1):
            if mask[i, j] > 127:
                end_idx = j
                break
        # print(start_idx, end_idx)
        # contour 검출
        if start_idx < end_idx:
            # print(i)
            mask[i, start_idx:end_idx] = 255
    pts1 = np.array(pts1)
    # pts1 = pts1.reshape(6, 1, 2)
    return mask, pts1

def get_center_of_mask(mask):
    # 흰색 영역의 픽셀 좌표 찾기
    white_pixels = np.where(mask == 255)  # 조건을 만족하는 픽셀 좌표 반환

    # 중심 좌표 계산
    center_y = np.mean(white_pixels[0]).astype(np.int16)  # x 좌표의 평균
    center_x = np.mean(white_pixels[1]).astype(np.int16)  # y 좌표의 평균
    return center_x, center_y

def get_homography_pokemon(image, pokemon, pts1, pts2):
    H, _ = cv2.findHomography(pts2, pts1, method=cv2.RANSAC) # pts1과 pts2의 행렬 주의 (N,1,2)
    result3 = cv2.warpPerspective(pokemon, H, (image.shape[1], image.shape[0]))
    return result3

image = cv2.imread("input.png")
mask, pts1 = get_facearea(image=image)
pokemon = cv2.imread("images/Bulbasaur/Bulbasaur_020.png")
pokemon = cv2.resize(pokemon, (512, 512))

center_of_mask = get_center_of_mask(mask)
pts2 = np.array([
    # [321, 359], # 입술 상
    # [321, 344], # 코 중앙
    [285, 150], # 이마(정수리)
    # [319, 380], # 입술 하
    # [203, 319], # 눈 좌하
    # [119, 343], # 입술 좌
    [296, 397], # 턱
    [122, 215], # 얼굴 좌상단
    [111, 342], # 얼굴 좌하단
    # [210, 249], # 눈 좌상
    # [410, 309], # 눈 우하
    # [436, 349], # 입술 우
    [426, 201], # 얼굴 우상단
    [454, 324], # 얼굴 우하단
    # [398, 234], # 눈 우상
])
result = get_homography_pokemon(image=image, pokemon=pokemon, pts1=pts1, pts2=pts2)
mixed = cv2.seamlessClone(result, image, mask, center_of_mask, cv2.NORMAL_CLONE)
import random
color_list = [(random.randrange(255), random.randrange(255), random.randrange(255))for i in range(50)]
for idx, point in enumerate(pts1):
    cv2.circle(image, point, 5, color_list[idx], 2)
for idx, point in enumerate(pts2):
    cv2.circle(pokemon, point, 5, color_list[idx], 2)

cv2.imwrite('mask.png', mask)
cv2.imwrite('pokemon.png', pokemon)

cv2.imshow('image', np.concatenate([image, pokemon], axis=1))

cv2.imshow('result', result)
cv2.imshow('mix', mixed)

cv2.waitKey()
cv2.destroyAllWindows()