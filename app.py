"""
animegan2
"""
import numpy as np

from PIL import Image
import torch


"""style mix"""
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# import numpy as np
import argparse
import gradio as gr

def get_facearea(image):
    # Predefined indices for specific landmarks of interest
    p = [
        10, 368, 397, 152, 172, 162
    ]

    # Empty list to store selected landmark coordinates
    pts1 = []  

    # Set up configuration options for face landmark detection
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)

    # Create the face detector from the options
    face_detector = vision.FaceLandmarker.create_from_options(options)

    # Retrieve the height, width, and channels of the input image
    h, w, _ = image.shape

    # Create an mp.Image object from the input image data
    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Detect face landmarks from the input image
    detection_result = face_detector.detect(image_mp)
    face_landmarks_list = detection_result.face_landmarks

    # Loop through the detected face landmarks
    for face_landmarks in face_landmarks_list:
        for idx, landmark in enumerate(face_landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            if idx in p:
                pts1.append((x, y))

    # Inner function to draw landmarks on the image
    def draw_landmarks_on_image(rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks using MediaPipe's drawing_utils
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )

        return annotated_image

    # Create a black mask image
    mask_rgb = np.zeros_like(image, dtype=np.uint8)  

    # Draw landmarks on the mask image
    mask_rgb = draw_landmarks_on_image(mask_rgb, detection_result)  

    # Convert the mask image to grayscale
    mask = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)  

    # Process each row of the grayscale mask image
    for i in range(512):
        start_idx = 0
        end_idx = -1

        # Find the start index of non-zero (white) pixel values
        for j in range(512):
            if mask[i, j] > 127:
                start_idx = j
                break

        # Find the end index of non-zero (white) pixel values
        for j in range(511, -1, -1):
            if mask[i, j] > 127:
                end_idx = j
                break

        # Fill the row of the mask with white pixels if a contour is detected
        if start_idx < end_idx:
            mask[i, start_idx:end_idx] = 255

    # Convert the selected landmark coordinates to a NumPy array
    pts1 = np.array(pts1)  

    # Reshape the array to a specific shape
    pts1 = pts1.reshape(6, 1, 2)  

    # Return the generated mask and the reshaped landmark coordinates
    return mask, pts1  

def get_center_of_mask(mask):
    # Find the coordinates of white pixels in the mask
    white_pixels = np.where(mask == 255)  # Returns the coordinates of pixels satisfying the condition

    # Calculate the center coordinates
    center_y = np.mean(white_pixels[0]).astype(np.int16)  # Average of x-coordinates
    center_x = np.mean(white_pixels[1]).astype(np.int16)  # Average of y-coordinates
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

def main(opt):
    model2 = torch.hub.load(
        "AK391/animegan2-pytorch:main",
        "generator",
        pretrained=True,
        device=opt.device,
        progress=False
    )

    face2paint = torch.hub.load(
        'AK391/animegan2-pytorch:main', 'face2paint', 
        size=512, device=opt.device, side_by_side=False
    )

    def greet(image):
        # Find the most similar pokemon.
        pokemon_path = "images/Mew/Mew_000.png" # path_list[rnd_idx]
        pokemon = cv2.imread(pokemon_path)
        pokemon = cv2.resize(pokemon, (512, 512))
        pokemon_name = "뮤츠"
        result_sentence = f"당신과 가장 닮은 포켓몬은 {pokemon_name}입니다!"

        # Make mask image from user image.
        mask, pts1 = get_facearea(image=image)

        # Convert from RGB to BGR (opencv use rgb channel).
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get center point of mask image.
        center_of_mask = get_center_of_mask(mask)

        # Apply homography from two point sets
        result = get_homography_pokemon(image=image, pokemon=pokemon, pts1=pts1)

        # Blend user image and pokemon image.
        # For more quility, i will use saemlessclone.
        mixed = cv2.seamlessClone(result, image, mask, center_of_mask, cv2.NORMAL_CLONE)
        
        # Convert from BGR to RGB
        color_coverted = cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB)

        # Convert from openCV2 to PIL for AnimeGAN2
        result=Image.fromarray(color_coverted)
        result = face2paint(model2, result)

        # Convert from openCV2 to PIL for AnimeGAN2
        return result_sentence, result

    demo = gr.Interface(
        fn=greet,
        inputs=[
            gr.inputs.Image(type="numpy", label="Input", shape=(512, 512), source=opt.source), 
            # gr.inputs.Radio(['True','False'], type="value", default='True', label='Run AnimeGAN')
            ],
        # inputs=["image"],
        outputs=[
            gr.outputs.Textbox(type = "text", label="가장 닮은 포켓몬은?"), 
            # gr.outputs.Image(type="numpy", label="Output1"), 
            # gr.outputs.Textbox(type = "text", label="Score"), 
            gr.outputs.Image(type="pil", label="Output"), 
            ],
        title="딥러닝 프로그래밍 기말 프로젝트",
        description="당신과 가장 닮은 포켓몬을 찾아보세요!",
        article = "<p style='text-align: center'><a href='https://github.com/hunsii/PokemonFaceSynthesis' target='_blank'>Github Repo by hunsii</a></p> <center></center></p>",
        # examples=[['input.png', 'True'], ['input.png', 'True']],
        # live=True,
    )
    demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps', help='Set which devices will you use. [cuda, cpu, mps]')
    parser.add_argument('--source', default='webcam', help='Set the way toupload image to model in gradio. [webcam, ]')
    opt = parser.parse_args()
    main(opt)
