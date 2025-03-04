import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import cv2
import onnxruntime as ort
import yaml
from yaml.loader import SafeLoader
import joblib
import pythreejs as p3
from IPython.display import display
import json 
import os
import trimesh
#from stl import mesh
import numpy as np
import plotly.graph_objects as go
# import pyrender
from PIL import Image
import streamlit.components.v1 as components

#html_path = "viewer.html"

# Load the HTML content
#with open(html_path, 'r') as f:
    #st.markdown("### #D model viewer")
    #html_content = f.read()

# Display the HTML content in Streamlit

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown('<p class="big-font">✈️SKYPULSE</p>', unsafe_allow_html=True) 
st.markdown('<p class="small-font">Fly Safe, Land Secure..!!</p>', unsafe_allow_html=True)

with open('data.yaml',mode='r') as f:
    data_yaml = yaml.load(f,Loader=SafeLoader)
    
labels = data_yaml['names']
#print(labels)

# Load the ONNX model , fault model
yolo = ort.InferenceSession('Model/weights/best.onnx')
Fault = joblib.load('Model/Wire_Fault.joblib')


def custom_css():
    st.markdown("""
        <style>
        .appview-container {
            background-image: linear-gradient(to right top, #b3dce3, #85c6dc, #57afd9, #2c96d5, #167bce);
        }
        .big-font {
            font-size:50px !important;
            font-weight: bold;
            color: white;
        }
        .small-font {
            font-family: monospace;
            font-weight: 200;
            font-style: italic;
            font-size:20px !important;
        }
        .normal-font {
            font-family: fangsong;
            font-weight: 800;
            font-size: 30px !important;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

custom_css()

# st.markdown('<p class="big-font">Hello, World!</p>', unsafe_allow_html=True)
# st.markdown('<p class="small-font">This is some small text.</p>', unsafe_allow_html=True)

def detect_dents_and_cracks(image):
    # Preprocess the image
    image = image.copy()
    row, col, d = image.shape

    # Convert image to a square image
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # Prepare the image blob
    INPUT_WH_YOLO = 640
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)

    # Perform inference using the ONNX model
    yolo.set_providers(['CPUExecutionProvider'])
    yolo_input_name = yolo.get_inputs()[0].name
    yolo_output_name = yolo.get_outputs()[0].name
    preds = yolo.run([yolo_output_name], {yolo_input_name: blob})[0]

    # Debugging information
    st.write(f"Predictions shape: {preds.shape}")
    

    detections = preds[0]
    boxes = []
    confidences = []
    classes = []

    # widht and height of the image (input_image)
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WH_YOLO
    y_factor = image_h/INPUT_WH_YOLO

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detection an object
        if confidence > 0.4:
            class_score = row[5:].max() # maximum probability from 20 objects
            class_id = row[5:].argmax() # get the index position at which max probabilty occur
        
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                # construct bounding from four values
                # left, top, width and height
                left = int((cx - 0.5*w)*x_factor)
                top = int((cy - 0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
            
                box = np.array([left,top,width,height])
            
                # append values into the list
                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)
            
    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()
    damage_locations = []
    for ind in index:
        # extract bounding box
        x,y,w,h = boxes_np[ind]
        bb_conf = int(confidences_np[ind]*100)
        classes_id = classes[ind]
        class_name = labels[classes_id]
    
        text = f'{class_name}: {bb_conf}%'
    
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,255,255),-1)
    
        cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_PLAIN,1.5,(0,0,0),2,cv2.LINE_AA)
        damage_locations.append((x, y, w, h, class_name))

    return image,damage_locations

# def render_3d():
    stl_path = 'AirplaneAllFiles/AirplaneForFreestl.stl'
    # Load the STL file using trimesh
    mesh = trimesh.load(stl_path)
    # Create a scene
    scene = pyrender.Scene()
    # Create a mesh node with the loaded STL file
    mesh_node = pyrender.Mesh.from_trimesh(mesh)
    # Add the mesh node to the scene
    scene.add(mesh_node)
    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    s = np.sqrt(2)/2
    camera_pose = np.array([
        [0.0, -s, s, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, s, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    # Set up the light
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
    scene.add(light, pose=camera_pose)
    # Render the scene
    r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
    color, _ = r.render(scene)
    image = Image.fromarray(color)
    st.image(image, caption='3D Model Render')


def suggest_repair_actions(damage_locations):
    st.write("### Suggested Repair Actions")
    for loc in damage_locations:
        x, y, w, h, class_name = loc
        st.write(f"- **{class_name}** detected at location (x={x}, y={y}, width={w}, height={h}):")
        if class_name == "dent":
            st.write("Inspect the dent for any cracks or paint damage:")
            st.write("- Use appropriate tools to repair the dent.")
            st.write("- Check for structural integrity post-repair.")
        elif class_name == "crack":
            st.write("Assess the extent of the crack:")
            st.write("- Determine if the crack is superficial or structural.")
            st.write("- Measure the length and depth of the crack.")
            st.write("Follow standard procedures to seal or replace the damaged part:")
            st.write("- Clean the area around the crack.")
            st.write("- Apply sealant or adhesive as per manufacturer's instructions.")
            st.write("- If necessary, replace the damaged part.")
            st.write("Perform a thorough inspection to ensure no further damage:")
            st.write("- Check adjacent areas for similar issues.")
            st.write("- Test the repaired area under normal operating conditions.")
        else:
            st.write("Follow appropriate maintenance procedures for the detected issue.")
        st.write("---")




if __name__ == "__main__":
  
    selected = option_menu(menu_title=None,
            options=['Damage Detection', 'Faulty Wire Detection'],
            icons =['activity','activity','info'],
            default_index=0,
            orientation='horizontal',
            menu_icon="cast")
    
    if selected == "Damage Detection":
        st.markdown('<p class="normal-font">Structural Integrity Assesment</p>', unsafe_allow_html=True)
        st.info('Upload Image to check any dents or cracks are present in image', icon="ℹ️")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
        # Read the uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        #create a table
            col1,col2 =st.columns([1, 1])
        # Display the original image
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)

        # Detect dents and cracks
            marked_image,damage_locations = detect_dents_and_cracks(image)

        # Display the marked image
            with col2:
                st.image(marked_image, caption="Image with Marked Faults", use_container_width=True)

            
            suggest_repair_actions(damage_locations)
            

    if selected == "Faulty Wire Detection" :
        st.markdown('<p class="normal-font">Electrical Fault Identification</p>', unsafe_allow_html=True)
        st.info('Enter data to check any faults', icon="ℹ️")
        v= st.text_input("Enter Voltage:")
        i = st.text_input("Enter Current:")
        r = st.text_input("Enter Resistance")

        def predict_wire_status(v, i, r):

            # Make prediction for the input data
            prediction = Fault.predict([[v, i, r]])

            if prediction[0] == 1:
                st.write('__Faulty Wire Detected__')
                st.write("Suggestions to deal with the faulty wire:")
                st.write("- Turn off the power supply before inspecting the wire.")
                st.write("- Inspect the wire for any visible damage, such as cuts, burns, or exposed wires.")
                st.write("- If the wire is damaged, cut out the damaged section and splice in a new section of wire.")
                st.write("- Use appropriate insulation materials to cover the spliced section.")
                st.write("- Test the repaired wire for continuity and insulation.")
            else:
                st.write('__No Fault__')
        
        if st.button('Predict'):
            st.write("Prediction:")
            result = predict_wire_status(v, i, r)
        
       

    if selected == "About":
        st.markdown('<h2 class="normal-font">About</h2>', unsafe_allow_html=True)

        # st.markdown("<h2 style='text-align: center;'>ABOUT</h2>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<p style='text-align: center;font-size:30px; font-family:Roboto'>Plane Checkups: Key to Safe Flights</p>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("""
        <div style="text-align: justify; font-family:times new roman"> 
            Aircraft maintenance and repair are integral components of the aviation industry, serving as the backbone of safety, reliability, and operational continuity. The meticulous assessment and meticulous repair of dents, damage, detection of faulty wires and wear on aircraft fuselage, wings, and other components are paramount to ensuring flight safety, regulatory compliance, and public confidence.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: justify; font-family:times new roman; padding-top: 10px"> 
            First and foremost, the safety of passengers, crew, and cargo is the primary concern in aviation. Any compromise to the structural integrity of an aircraft, no matter how minor, poses a potential threat to safety. Damage, such as dents or structural deformities, can disrupt airflow, compromise aerodynamics, and weaken critical structural elements, increasing the risk of catastrophic failures during flight.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: justify; font-family:times new roman; padding-top: 10px"> 
            Similarly, faulty wiring poses a significant safety risk, as it can lead to electrical malfunctions within critical aircraft systems. These malfunctions may result in system failures, in-flight emergencies, or even fires, jeopardizing the safety of passengers, crew, and the aircraft itself.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: justify; font-family:times new roman; padding-top: 10px"> 
            
Thus, thorough assessments and repairs are essential to maintaining the airworthiness of aircraft and safeguarding against potential accidents.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<h4 style='text-align: center;font-size:30px'>CONTRIBUTORS</h4>", unsafe_allow_html=True)
        st.markdown("""<div style="text-align: center; padding: 5px;"><a href="https://linkedin.com/in/soumikisonline" style="text-decoration: none;">Soumik Saha</a></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="text-align: center; padding: 5px;"><a href="https://www.linkedin.com/in/ayushi-gupta-72a38121b" style="text-decoration: none;">Ayushi Gupta</a></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="text-align: center; padding: 5px;"><a href="https://www.linkedin.com/in/souvik-dey-80b033241/" style="text-decoration: none;">Souvik Dey</a></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="text-align: center; padding: 5px;"><a href="https://www.linkedin.com/in/pankaj-goel-30195720a/" style="text-decoration: none;">Pankaj Goel</a></div>""", unsafe_allow_html=True)
        st.markdown("""<div style="text-align: center; padding: 5px;"><a href="https://www.linkedin.com/in/bhagyasri-u/" style="text-decoration: none;">Bhagyasri Uddandam</a></div>""", unsafe_allow_html=True)
        st.markdown("____")
