import numpy as np
import gradio as gr
from PIL import Image
import torch



"""
animegan2
"""
model2 = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained=True,
    device="mps",
    progress=False
)

face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint', 
    size=512, device="mps",side_by_side=False
)

"""style mix"""
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import numpy as np

def get_facearea(image):
    p = [
        10, 368, 397, 152, 172, 162
    ]
    pts1 = []
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    face_detector = vision.FaceLandmarker.create_from_options(options)

    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape

    # STEP 3: Load the input image.
    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # STEP 4: Detect face landmarks from the input image.
    detection_result = face_detector.detect(image_mp)
    face_landmarks_list = detection_result.face_landmarks
    face_blendshapes_list = detection_result.face_blendshapes

    # Loop through the detected faces to visualize.
    for face_landmarks in face_landmarks_list:
        for idx, landmark in enumerate(face_landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
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
    pts1 = pts1.reshape(6, 1, 2)
    return mask, pts1

def get_center_of_mask(mask):
    # 흰색 영역의 픽셀 좌표 찾기
    white_pixels = np.where(mask == 255)  # 조건을 만족하는 픽셀 좌표 반환

    # 중심 좌표 계산
    center_y = np.mean(white_pixels[0]).astype(np.int16)  # x 좌표의 평균
    center_x = np.mean(white_pixels[1]).astype(np.int16)  # y 좌표의 평균
    return center_x, center_y

def get_homography_pokemon(image, pokemon, pts1):
    pts2 = np.array([
        [[198,  83]], 
        [[294, 162]],
        [[144, 140]],
        [[170, 206]],
        [[288, 110]], 
        [[225, 220]],
    ])

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC) # pts1과 pts2의 행렬 주의 (N,1,2)
    result3 = cv2.warpPerspective(pokemon, H, (image.shape[1], image.shape[0]))
    return result3

def greet(image):
    # print(image)
    # Get Cos similiarity
    # import glob
    # path_list = glob.glob("images/*/*.png")
    # rnd_idx = np.random.randint(0, 2000)
    pokemon_path = "images/Mew/Mew_000.png" # path_list[rnd_idx]
    pokemon = cv2.imread(pokemon_path)
    pokemon = cv2.resize(pokemon, (512, 512))

    mask, pts1 = get_facearea(image=image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    center_of_mask = get_center_of_mask(mask)
    result = get_homography_pokemon(image=image, pokemon=pokemon, pts1=pts1)
    mixed = cv2.seamlessClone(result, image, mask, center_of_mask, cv2.NORMAL_CLONE)
    # convert from BGR to RGB
    color_coverted = cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB)

    # convert from openCV2 to PIL
    result=Image.fromarray(color_coverted)
    # print(flags)
    # print(type(flags))
    # if flags == 'True':
    result = face2paint(model2, result)

    # Style mixing
    return f"당신과 가장 닮은 포켓몬은 뮤츠입니다!", cv2.cvtColor(pokemon, cv2.COLOR_BGR2RGB), result

demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.inputs.Image(type="numpy", label="Input", shape=(512, 512), source="webcam"), 
        # gr.inputs.Radio(['True','False'], type="value", default='True', label='Run AnimeGAN')
        ],
    # inputs=["image"],
    outputs=[
        gr.outputs.Textbox(type = "text", label="가장 닮은 포켓몬은?"), 
        gr.outputs.Image(type="numpy", label="Output1"), 
        # gr.outputs.Textbox(type = "text", label="Score"), 
        gr.outputs.Image(type="pil", label="Output2"), 
        ],
    title="딥러닝 프로그래밍 기말 프로젝트",
    description="당신과 가장 닮은 포켓몬을 찾아보세요!",
    # article = "<p style='text-align: center'><a href='https://github.com/jjeamin/anime_style_transfer_pytorch' target='_blank'>Github Repo by jjeamin</a></p> <center><img src='https://visitor-badge.glitch.me/badge?page_id=jjeamin_arcane_st' alt='visitor badge'></center></p>",
    # examples=[['input.png', 'True'], ['input.png', 'True']],
    # live=True,
)
demo.launch()
