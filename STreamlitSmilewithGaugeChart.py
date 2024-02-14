import cv2
import numpy as np
import pandas as pd
import time
import altair as alt
import streamlit as st
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Initialize session state
if "component_states" not in st.session_state:
    st.session_state["component_states"] = {}

# Load the face and smile classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# Load the pre-trained model for smile detection
model = load_model(model.h5')

df = []
cap = cv2.VideoCapture(0)
# Initialize the DataFrame for storing the face IDs and happiness indices
df = pd.DataFrame(columns=['Timestamp', 'Face ID', 'Happiness Index', 'Location'])


run_detection = False
last_captured_frame = None
happiness_index = None


def detect_smile(frame, df,location):
    global happiness_index
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Define the happiness index
    happiness_index = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (32, 32))
        roi = roi / 255.0
        roi = roi[np.newaxis, ...]
        smile_prob = model.predict(roi)[0]
        if smile_prob > 0.75:
            # Calculate the happiness index based on the smile probability
            happiness_index = smile_prob * 100
            # Generate a unique face ID based on the face coordinates with timestamp
            face_id = f"{x}-{y}-{w}-{h}-{int(time.time())}"
            # Append the face ID and happiness index to the DataFrame
            new_row = {'Timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 'Face ID': face_id, 'Happiness Index': happiness_index, 'Location': location}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Save the data to CSV file without truncating the previous data
            df.to_csv(happiness_details.csv", mode='a', index=False, header=False)

            label = "Thanks for Smiling!"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        else:
            label = "No Smile Detected!"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # Display the frame with the smile detection and happiness index
        cv2.putText(frame, f"Happiness Index: {float(happiness_index):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Smile Detector", frame)

    return frame, df


def generate_gauge_chart(happiness_index):
    happiness_value = float(happiness_index)  # Extract scalar value from the NumPy array
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=happiness_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Happiness Index", 'font': {'size': 24}},
        delta={'reference': 90, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "lavenderblush"},
            'bgcolor': "lavender",
            'borderwidth': 1,
            'bordercolor': "gray",
            'steps': [
                {'range': [70, 90], 'color': 'turquoise'},
                {'range': [90, 100], 'color': 'seagreen'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95}}))

    fig.update_layout(paper_bgcolor="lavender", font={'color': "teal", 'family': "Arial"})

    return fig





def categorize_happiness(happiness_index):
    if 90 <= happiness_index <= 100:
        return "Very Happy"
    elif 70 <= happiness_index < 90:
        return "Satisfied"
    elif 60 <= happiness_index < 70:
        return "Neutral"
    else:
        return "Unsatisfied"


def get_color(category_counts):
    colors = {
        'darkgreen': ['Very Happy'],
        'lightgreen': ['Happy'],
        'orange': ['Neutral'],
        'red': ['Unsatisfied', 'Unhappy']
    }
    return [next((color for color, categories in colors.items() if category in categories), 'black')
            for category in category_counts.index]


def main():
    st.set_page_config(page_title="Smile Detector", page_icon=":smiley:")

    # Create a title for the app
    st.title("Happiness Index Detection")
    activities = ["Selfie Portal", "About Us"]
    choice = st.sidebar.selectbox("Select Activity", activities)
	
	# Add an input field for location
    location = st.text_input("Enter your location:")

    if choice == "Selfie Portal":
        st.header("Smile Please")
        st.write("Click on Start to use the camera and detect your happiness index")

        if st.button("Start"):
            global last_captured_frame, run_detection, happiness_index
            # Create a DataFrame to store the happiness index
            df = pd.DataFrame(columns=['Timestamp', 'Face ID', 'Happiness Index', 'Location'])

            # Open the default camera and start the smile detection
            cap = cv2.VideoCapture(0)
            run_detection = True

            while run_detection:
                ret, frame = cap.read()

                # Call the detect_smile function to detect smile and calculate happiness index
                frame, df = detect_smile(frame, df, location)

                # Store the last captured frame
                last_captured_frame = frame.copy()

                # Display the webcam feed and happiness index
                cv2.imshow('Smile Detector', frame)

                # Check if the user wants to stop the detection
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    run_detection = False

            # Release the camera and close the window
            cap.release()
            cv2.destroyAllWindows()

            # Ensure the "Happiness Index" column has a numeric data type
            df['Happiness Index'] = df['Happiness Index'].astype(float)

            # Calculate statistics
            average_happiness = df['Happiness Index'].mean()
            max_happiness = df['Happiness Index'].max()
            #min_happiness = df['Happiness Index'].min()
            #num_distinct_smiles = df['Face ID'].nunique()

            # Categorize happiness based on the happiness index
            df['Happiness Category'] = df['Happiness Index'].apply(categorize_happiness)

            #st.subheader("KPIs")
            
            #st.write(f"Maximum Happiness Index: {max_happiness:.2f}")
            #st.write(f"Minimum Happiness Index: {min_happiness:.2f}")
            #st.write(f"Number of Distinct Smiles Detected: {num_distinct_smiles}")


            # Display the happiness index gauge chart
            st.subheader("Happiness Index Gauge Chart")
            fig = generate_gauge_chart(happiness_index)
            st.plotly_chart(fig)

            # Display the last captured frame
            if last_captured_frame is not None:
                st.subheader("Last Captured Frame")
                last_captured_frame_rgb = cv2.cvtColor(last_captured_frame, cv2.COLOR_BGR2RGB)
                st.image(last_captured_frame_rgb, caption="User's Smiling Face")

            # Display the happiness index using KPI card
            st.subheader("What does your happiness index tells?")
            #st.info(f"{max_happiness:.2f}")
            st.write(f"Maximum Happiness Index: {max_happiness:.2f}")
            st.write(f"Average Happiness Index: {average_happiness:.2f}")

            # Display the thank you message
            st.success("Thanks for smiling, have a good day!")

        elif choice == "About Us":
            st.subheader("About this app")
            html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
                                        <h4 style="color:white;text-align:center;">
                                        Real time Smile Detection application to detect Happiness Index.</h4>
                                        </div>
                                        </br>"""
            st.markdown(html_temp_about1, unsafe_allow_html=True)

            html_temp4 = """
                                 		<div style="background-color:#98AFC7;padding:10px">
                                 		<h4 style="color:white;text-align:center;">This Application is developed  using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose.</h4>
                                 		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                 		</div>
                                 		<br></br>
                                 		<br></br>"""

            st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
