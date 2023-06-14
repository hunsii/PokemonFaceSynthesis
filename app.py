"""
animegan2
"""
import numpy as np

from PIL import Image
import torch

import timm

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
import glob

from torchvision import transforms
import ast

def get_pokemon_point_list():
    """
    Retrieves the data points of Pokémon from text files and returns them as a dictionary.

    Returns:
        data_dict (dict): A dictionary containing Pokémon names as keys and their corresponding data points as values.
    """
    l = glob.glob("points/*.txt")
    data_dict = {}
    data = ""
    for path in l:
        with open(path, "r") as f:
            data+= f.read().strip()
    datas = data.split("-"*19)
    for data_dump in datas:
        # print(data_dump)
        # print('-'*80)
        lines = data_dump.strip().splitlines()
        if len(lines) == 0:
            continue
        basename = lines[0].split("\\")[-1]
        # print(f"basename: {basename}")

        points = []
        # (x, y) 형태의 숫자 데이터를 (x, y) 튜플로 파싱하여 리스트에 저장합니다.
        for idx in [1, 4, 6, 5, 2, 3]:
            line = lines[idx]
            line = line.replace("(", "").replace(")", "")
            points.append([int(v) for v in line.split(", ")])
        
        # 파싱된 데이터를 NumPy 배열로 변환합니다.
        numpy_array = np.array(points)
        # print(numpy_array)

        # data_dict = {}
        data_dict[basename.split('_')[0]] = numpy_array
    return data_dict

def get_facearea(image):
    """
    Detects and extracts the face area from an input image.

    Args:
        image (ndarray): An input image represented as a NumPy array with RGB channel.

    Returns:
        mask (ndarray): A binary mask representing the detected face area.
        pts1 (ndarray): Selected landmark coordinates of the detected face.
    """
    
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
    """
    Calculates the center coordinates of white pixels in a binary mask.

    Args:
        mask (ndarray): A binary mask image.

    Returns:
        center_x (int): The x-coordinate of the center.
        center_y (int): The y-coordinate of the center.
    """
    # Find the coordinates of white pixels in the mask
    white_pixels = np.where(mask == 255)  # Returns the coordinates of pixels satisfying the condition

    # Calculate the center coordinates
    center_y = np.mean(white_pixels[0]).astype(np.int16)  # Average of x-coordinates
    center_x = np.mean(white_pixels[1]).astype(np.int16)  # Average of y-coordinates
    return center_x, center_y

def get_homography_pokemon(image, pokemon, pts1, pts2):
    """
    Warps a Pokémon image onto an input image using homography transformation.

    Args:
        image (ndarray): An input image represented as a NumPy array.
        pokemon (ndarray): A Pokémon image represented as a NumPy array.
        pts1 (ndarray): Selected landmark coordinates of the detected face.
        pts2 (ndarray): Landmark coordinates of the Pokémon image.

    Returns:
        result (ndarray): The input image with the Pokémon image warped onto it.

    """
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC) # pts1과 pts2의 행렬 주의 (N,1,2)
    result3 = cv2.warpPerspective(pokemon, H, (image.shape[1], image.shape[0]))
    return result3

def main(opt):
    # Load pretrained model from path
    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.load_state_dict(torch.load("saved_model_b0.pth", map_location=opt.device))
    model.eval()

    # Set transform using efficientnet_b0
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 파일에서 문자열 읽기
    with open("pokemon_mapping.txt", "r") as file:
        data = file.readlines()[0].split('=',maxsplit=2)[-1]#.split('\n')[0]

    # 문자열을 dict 형태로 변환
    pokemon_mapping_dict = ast.literal_eval(data)

    # Load animegan2 model from pytorch hub.
    model2 = torch.hub.load(
        "AK391/animegan2-pytorch:main",
        "generator",
        pretrained=True,
        device=opt.device,
        progress=False
    )

    # Load face2paint model from pytorch hub.
    face2paint = torch.hub.load(
        'AK391/animegan2-pytorch:main', 'face2paint', 
        size=512, device=opt.device, side_by_side=False
    )

    # Get pokemon point list
    pokemon_point_list = get_pokemon_point_list()


    # This fuction work when user sent image.
    def greet(image):
        # Convert from openCV2 to PIL for AnimeGAN2
        image_pil = Image.fromarray(image)
        image_transformed = transform(image_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_transformed)
            
        # 가장 높은 값을 가진 클래스 선택
        _, predicted = torch.max(outputs, 1)

        pokemon_name = pokemon_mapping_dict[str(predicted.detach().numpy()[0])]

        pokemon = cv2.imread(f"images/{pokemon_name}/{pokemon_name}_000.png")
        pokemon = cv2.resize(pokemon, (512, 512))

        result_sentence = f"당신과 가장 닮은 포켓몬은 {pokemon_name}입니다!"

        # Make mask image and landmarks from user image.
        mask, pts1 = get_facearea(image=image)

        # Get center point of mask image.
        center_of_mask = get_center_of_mask(mask)

        # Get pokemon landmarks
        pts2 = pokemon_point_list[pokemon_name]

        # Apply homography from two point sets
        result = get_homography_pokemon(image=image, pokemon=pokemon, pts1=pts1, pts2=pts2)

        # Blend user image and pokemon image.
        # For more quility, i will use saemlessclone.
        mixed = cv2.seamlessClone(result, image, mask, center_of_mask, cv2.NORMAL_CLONE)
        
        # Convert from BGR to RGB
        # color_coverted = cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB)

        # Convert from openCV2 to PIL for AnimeGAN2
        result = Image.fromarray(mixed)
        result = face2paint(model2, result)

        # Convert from openCV2 to PIL for AnimeGAN2
        return result_sentence, result

    # Make gradio Web interface
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
        examples=[['26000.png'], ['33002.png'], ['65062.png'], ['65065.png']],
        # live=True,
    )
    demo.launch(share=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps', help='Set which devices will you use. [cuda, cpu, mps]')
    parser.add_argument('--source', default='webcam', help='Set the way toupload image to model in gradio. [webcam, ]')
    opt = parser.parse_args()
    main(opt)
