import streamlit as st
import os
import io
import boto3
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torch
import torchvision
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import numpy as np
from io import StringIO, BytesIO


# Creating the Layout of the App
st.set_page_config(layout="wide")
row1_1, row1_2 = st.columns(2)
col1, col2 = st.columns(2)


def get_user_input():
    uploaded_file = col1.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # user uploads an image
    if uploaded_file is not None:
        user_file_path = os.path.join("images", uploaded_file.name)
        with open(user_file_path, "wb") as user_file:
            user_file.write(uploaded_file.getbuffer())

    # print(user_file_path)
    # default image, coffee shop
    else:
        user_file_path = os.path.join("images", "cafe.jpg")

    return user_file_path


#     except:
#         print("bugugugu")
#         st.error('This is an error', icon="🚨")
#         st.error("Please enter a valid input")
#         st.write("Invalid Input: image must be of type jpg, jpeg, or png")
#         return None


# IMPORTANT: Cache the Amazon Rekognition API call to prevent computation on every rerun
# When you mark a function with Streamlit’s cache annotation,
# it tells Streamlit that whenever the function is called it should
# check the name of the function, actual code that makes up the body of the function,
# and input parameters that you called the function with.
# If this is the first time Streamlit has seen those three items,
# with those exact values, and in # that exact combination,
# it runs the function and stores the result in a local cache.


@st.cache_data
def detect_labels(filename):
    client = boto3.client("rekognition")
    image = Image.open(filename)

    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    image_binary = stream.getvalue()

    response = client.detect_labels(Image={"Bytes": image_binary}, MaxLabels=100)
    return response


@st.cache_data
def visualize_output(filename, class_name, predictions_df):
    """
    Given an input image and predictions, returns the annotated image,
    number of instances of the label detected, and the average confidence for that label.
    """

    # read the saved input image
    img = read_image(filename)
    predictions_df = predictions_df.loc[predictions_df["label"] == class_name]
    instances = len(predictions_df)
    average_confidence = predictions_df["confidence"].mean()
    for index, row in predictions_df.iterrows():
        score = row["confidence"]
        x1 = row["x1"]
        x2 = row["x2"]
        y1 = row["y1"]
        y2 = row["y2"]
        box = [x1, y1, x2, y2]

        label = class_name + ", score: " + str(round(score, 2))

        # draw bounding box and fill color
        box = torch.tensor(box)
        box = box.unsqueeze(0)
        img = draw_bounding_boxes(
            img, box, width=5, labels=[label], font_size=24, colors="red"
        )

    # transform this image to PIL image
    img = torchvision.transforms.ToPILImage()(img)

    # display output
    return img, instances, average_confidence


def process_response(response, filename):
    """
    Processes the recognition response, by creating a pandas dataframe with the output predictions
    """

    labels = []
    predictions = {}
    data = []
    image = Image.open(filename)
    imgWidth, imgHeight = image.size
    for detected_label in response["Labels"]:
        instances = len(detected_label["Instances"])

        if instances:
            name = detected_label["Name"]
            confidence = detected_label["Confidence"]
            labels.append(name)

            predictions[name] = {
                "predictions": detected_label["Instances"],
            }

    for label in predictions:
        for prediction in predictions[label]["predictions"]:
            score = round(prediction["Confidence"], 2)
            box = prediction["BoundingBox"]

            left = imgWidth * box["Left"]
            top = imgHeight * box["Top"]
            width = imgWidth * box["Width"]
            height = imgHeight * box["Height"]

            x1 = left
            x2 = left + width
            y1 = top
            y2 = top + height
            row = {
                "label": label,
                "confidence": score,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
            }
            data.append(row)
    df = pd.DataFrame.from_records(data)
    return df


def convert_df(df):
    """
    Takes in a pandas dataframe, converts that dataframe to a csv and returns the output
    """

    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


def main():
    row1_1.title("Rekognition Object Detection Demo")
    row1_2.write(
        """
        ##
        Rekognition Label Detection detects instances of real-world entities within an image (JPEG or PNG) provided as
        input. This includes objects like flower, tree, and table; events like wedding, graduation, and birthday party; 
        and concepts like landscape, evening,
        and nature.
        """
    )
    imageLocation = col2.empty()
    input_image_path = get_user_input()
    if input_image_path:
        try:
            imageLocation.image(input_image_path, use_column_width=True)
        except:
            st.error("Invalid Input: image must be of type jpg, jpeg, or png")
            input_image_path = None

    selectboxLocation = col1.empty()
    selectboxLocation.selectbox("Visualize Rekognition Results", ["Select"])

    instance_location = col1.empty()
    confidence_location = col1.empty()

    if input_image_path is not None:
        response = detect_labels(input_image_path)

        df = process_response(response, input_image_path)
        if not df.empty:
            labels = list(df["label"].unique())

            option = selectboxLocation.selectbox(
                "Visualize Rekognition Results", ["Select"] + labels
            )

            if option != "Select":
                image, instances, average_confidence = visualize_output(
                    input_image_path, option, df
                )
                instance_location.write(
                    f"Average Confidence: {round(average_confidence, 2)}"
                )
                confidence_location.write(f"Instances: {instances}")

                imageLocation.image(image)
        else:
            st.warning(
                "No Amazon Rekognition Output: Amazon Rekognition did not identify any valid objects. Please see the Amazon Rekognition documentation for more information"
            )

        csv = convert_df(df)
        col1.download_button(
            label="Download Output as CSV",
            data=csv,
            file_name="large_df.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
