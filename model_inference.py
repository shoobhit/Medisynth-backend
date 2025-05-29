
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import os
import logging
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (224, 224)
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
NUM_CLASSES = len(CLASS_NAMES)

# Disease descriptions
DISEASE_DESCRIPTIONS = {
    'Atelectasis': {
        'description': 'Atelectasis is a condition where part or all of a lung collapses, leading to reduced or absent air in the affected area.',
        'location_specific': {
            'left_upper': 'Focal opacity in the left upper lobe suggestive of atelectasis, indicating partial lung collapse.',
            'left_middle': 'Focal opacity in the left middle lobe suggestive of atelectasis, indicating partial lung collapse.',
            'left_lower': 'Focal opacity in the left lower lobe suggestive of atelectasis, indicating partial lung collapse.',
            'right_upper': 'Focal opacity in the right upper lobe suggestive of atelectasis, indicating partial lung collapse.',
            'right_middle': 'Focal opacity in the right middle lobe suggestive of atelectasis, indicating partial lung collapse.',
            'right_lower': 'Focal opacity in the right lower lobe suggestive of atelectasis, indicating partial lung collapse.',
            'central': 'Opacity in the central lung fields suggestive of atelectasis, possibly bilateral.',
            'unknown': 'Focal opacity suggestive of atelectasis, location unspecified.'
        },
        'symptoms': 'Symptoms may include shortness of breath, chest pain, and coughing. In mild cases, it may be asymptomatic.',
        'causes': 'Common causes include obstruction of the airways (e.g., by mucus or a foreign object), lung compression (e.g., by fluid or tumors), or surgery affecting lung expansion.',
        'implications': 'If untreated, atelectasis can lead to infection or reduced oxygen levels. Further evaluation with CT or bronchoscopy is recommended.'
    },
    'Cardiomegaly': {
        'description': 'Cardiomegaly refers to an enlarged heart, often detected on imaging as an increased heart size.',
        'location_specific': {
            'central': 'Enlarged cardiac silhouette in the central region, consistent with cardiomegaly.',
            'unknown': 'Enlarged cardiac silhouette, consistent with cardiomegaly.'
        },
        'symptoms': 'Symptoms may include fatigue, shortness of breath, swelling in the legs, and irregular heartbeats, though it can be asymptomatic.',
        'causes': 'Causes include hypertension, heart valve disease, cardiomyopathy, or chronic lung disease leading to increased heart workload.',
        'implications': 'Cardiomegaly may indicate underlying heart disease requiring further cardiac evaluation, such as echocardiography.'
    },
    'Consolidation': {
        'description': 'Consolidation occurs when lung tissue fills with liquid instead of air, often due to infection or inflammation.',
        'location_specific': {
            'left_upper': 'Consolidation in the left upper lobe, indicating fluid-filled lung tissue.',
            'left_middle': 'Consolidation in the left middle lobe, indicating fluid-filled lung tissue.',
            'left_lower': 'Consolidation in the left lower lobe, indicating fluid-filled lung tissue.',
            'right_upper': 'Consolidation in the right upper lobe, indicating fluid-filled lung tissue.',
            'right_middle': 'Consolidation in the right middle lobe, indicating fluid-filled lung tissue.',
            'right_lower': 'Consolidation in the right lower lobe, indicating fluid-filled lung tissue.',
            'central': 'Bilateral consolidation in the central lung fields, indicating fluid-filled lung tissue.',
            'unknown': 'Consolidation indicating fluid-filled lung tissue, location unspecified.'
        },
        'symptoms': 'Symptoms include fever, cough with sputum, chest pain, and difficulty breathing.',
        'causes': 'Most commonly caused by pneumonia, but can also result from pulmonary edema or hemorrhage.',
        'implications': 'Consolidation suggests an active process, often infectious, requiring prompt medical evaluation and possibly antibiotics.'
    },
    'Edema': {
        'description': 'Pulmonary edema is the accumulation of fluid in the lung’s air sacs, impairing gas exchange.',
        'location_specific': {
            'left_upper': 'Hazy opacities in the left upper lobe suggestive of pulmonary edema.',
            'left_middle': 'Hazy opacities in the left middle lobe suggestive of pulmonary edema.',
            'left_lower': 'Hazy opacities in the left lower lobe suggestive of pulmonary edema.',
            'right_upper': 'Hazy opacities in the right upper lobe suggestive of pulmonary edema.',
            'right_middle': 'Hazy opacities in the right middle lobe suggestive of pulmonary edema.',
            'right_lower': 'Hazy opacities in the right lower lobe suggestive of pulmonary edema.',
            'central': 'Bilateral hazy opacities in the central lung fields, consistent with pulmonary edema.',
            'unknown': 'Hazy opacities suggestive of pulmonary edema, location unspecified.'
        },
        'symptoms': 'Symptoms include severe shortness of breath, wheezing, coughing up frothy sputum, and a feeling of suffocation.',
        'causes': 'Typically caused by congestive heart failure, but can also result from kidney failure, high-altitude exposure, or acute lung injury.',
        'implications': 'Pulmonary edema is a medical emergency requiring immediate treatment to address the underlying cause and support breathing.'
    },
    'Effusion': {
        'description': 'Pleural effusion is the buildup of fluid between the lung and chest wall in the pleural space.',
        'location_specific': {
            'left_upper': 'Blunting of the left costophrenic angle suggestive of pleural effusion.',
            'left_middle': 'Fluid layering in the left pleural space suggestive of pleural effusion.',
            'left_lower': 'Blunting of the left costophrenic angle suggestive of pleural effusion.',
            'right_upper': 'Blunting of the right costophrenic angle suggestive of pleural effusion.',
            'right_middle': 'Fluid layering in the right pleural space suggestive of pleural effusion.',
            'right_lower': 'Blunting of the right costophrenic angle suggestive of pleural effusion.',
            'central': 'Bilateral fluid layering suggestive of pleural effusion.',
            'unknown': 'Fluid layering suggestive of pleural effusion, location unspecified.'
        },
        'symptoms': 'Symptoms may include chest pain, shortness of breath, and a dry cough, though small effusions may be asymptomatic.',
        'causes': 'Causes include infections (e.g., pneumonia), heart failure, malignancy, or autoimmune diseases.',
        'implications': 'The underlying cause must be identified, as effusion may require drainage or treatment of the primary condition.'
    },
    'Emphysema': {
        'description': 'Emphysema is a chronic lung condition characterized by damaged air sacs, leading to reduced lung elasticity.',
        'location_specific': {
            'left_upper': 'Hyperlucency in the left upper lobe suggestive of emphysema, indicating air trapping.',
            'left_middle': 'Hyperlucency in the left middle lobe suggestive of emphysema, indicating air trapping.',
            'left_lower': 'Hyperlucency in the left lower lobe suggestive of emphysema, indicating air trapping.',
            'right_upper': 'Hyperlucency in the right upper lobe suggestive of emphysema, indicating air trapping.',
            'right_middle': 'Hyperlucency in the right middle lobe suggestive of emphysema, indicating air trapping.',
            'right_lower': 'Hyperlucency in the right lower lobe suggestive of emphysema, indicating air trapping.',
            'central': 'Bilateral hyperlucency suggestive of emphysema, indicating air trapping.',
            'unknown': 'Hyperlucency suggestive of emphysema, location unspecified.'
        },
        'symptoms': 'Symptoms include chronic shortness of breath, wheezing, and fatigue, worsening over time.',
        'causes': 'Primarily caused by long-term smoking, but also by air pollution or genetic factors (e.g., alpha-1 antitrypsin deficiency).',
        'implications': 'Emphysema is irreversible but manageable with lifestyle changes, medications, or oxygen therapy.'
    },
    'Fibrosis': {
        'description': 'Pulmonary fibrosis involves scarring of lung tissue, leading to stiff lungs and reduced oxygen transfer.',
        'location_specific': {
            'left_upper': 'Reticular opacities in the left upper lobe suggestive of pulmonary fibrosis.',
            'left_middle': 'Reticular opacities in the left middle lobe suggestive of pulmonary fibrosis.',
            'left_lower': 'Reticular opacities in the left lower lobe suggestive of pulmonary fibrosis.',
            'right_upper': 'Reticular opacities in the right upper lobe suggestive of pulmonary fibrosis.',
            'right_middle': 'Reticular opacities in the right middle lobe suggestive of pulmonary fibrosis.',
            'right_lower': 'Reticular opacities in the right lower lobe suggestive of pulmonary fibrosis.',
            'central': 'Bilateral reticular opacities suggestive of pulmonary fibrosis.',
            'unknown': 'Reticular opacities suggestive of pulmonary fibrosis, location unspecified.'
        },
        'symptoms': 'Symptoms include progressive shortness of breath, dry cough, fatigue, and clubbing of fingers.',
        'causes': 'Causes include autoimmune diseases, environmental exposures (e.g., asbestos), or idiopathic factors.',
        'implications': 'Fibrosis is progressive and requires management to slow progression and improve quality of life.'
    },
    'Hernia': {
        'description': 'A diaphragmatic hernia is a defect in the diaphragm allowing abdominal organs to protrude into the chest cavity.',
        'location_specific': {
            'left_upper': 'Abnormal contour in the left upper chest suggestive of diaphragmatic hernia.',
            'left_middle': 'Abnormal contour in the left middle chest suggestive of diaphragmatic hernia.',
            'left_lower': 'Abnormal contour in the left lower chest suggestive of diaphragmatic hernia.',
            'right_upper': 'Abnormal contour in the right upper chest suggestive of diaphragmatic hernia.',
            'right_middle': 'Abnormal contour in the right middle chest suggestive of diaphragmatic hernia.',
            'right_lower': 'Abnormal contour in the right lower chest suggestive of diaphragmatic hernia.',
            'central': 'Abnormal contour in the central chest suggestive of diaphragmatic hernia.',
            'unknown': 'Abnormal contour suggestive of diaphragmatic hernia, location unspecified.'
        },
        'symptoms': 'Symptoms may include chest pain, difficulty breathing, and gastrointestinal issues like reflux.',
        'causes': 'Can be congenital or acquired due to trauma or surgery.',
        'implications': 'Surgical repair may be needed, depending on severity and symptoms.'
    },
    'Infiltration': {
        'description': 'Infiltration refers to abnormal substances (e.g., fluid, cells) in the lungs, appearing as diffuse opacities on X-rays.',
        'location_specific': {
            'left_upper': 'Diffuse opacities in the left upper lobe suggestive of infiltration.',
            'left_middle': 'Diffuse opacities in the left middle lobe suggestive of infiltration.',
            'left_lower': 'Diffuse opacities in the left lower lobe suggestive of infiltration.',
            'right_upper': 'Diffuse opacities in the right upper lobe suggestive of infiltration.',
            'right_middle': 'Diffuse opacities in the right middle lobe suggestive of infiltration.',
            'right_lower': 'Diffuse opacities in the right lower lobe suggestive of infiltration.',
            'central': 'Bilateral diffuse opacities suggestive of infiltration.',
            'unknown': 'Diffuse opacities suggestive of infiltration, location unspecified.'
        },
        'symptoms': 'Symptoms vary but may include cough, fever, and shortness of breath, depending on the cause.',
        'causes': 'Common causes include infections, pulmonary edema, or allergic reactions.',
        'implications': 'Further diagnostic workup is needed to determine the cause and appropriate treatment.'
    },
    'Mass': {
        'description': 'A lung mass is a growth in the lung, which may be benign or malignant.',
        'location_specific': {
            'left_upper': 'Well-defined mass in the left upper lobe, suggestive of a lung mass.',
            'left_middle': 'Well-defined mass in the left middle lobe, suggestive of a lung mass.',
            'left_lower': 'Well-defined mass in the left lower lobe, suggestive of a lung mass.',
            'right_upper': 'Well-defined mass in the right upper lobe, suggestive of a lung mass.',
            'right_middle': 'Well-defined mass in the right middle lobe, suggestive of a lung mass.',
            'right_lower': 'Well-defined mass in the right lower lobe, suggestive of a lung mass.',
            'central': 'Mass in the central lung fields, suggestive of a lung mass.',
            'unknown': 'Well-defined mass suggestive of a lung mass, location unspecified.'
        },
        'symptoms': 'Symptoms may include cough, chest pain, weight loss, or hemoptysis, though small masses may be asymptomatic.',
        'causes': 'Causes include lung cancer, benign tumors, or infections like granulomas.',
        'implications': 'Urgent evaluation, possibly with biopsy, is required to rule out malignancy.'
    },
    'No Finding': {
        'description': 'No abnormalities were detected in the X-ray, indicating a normal lung appearance.',
        'location_specific': {
            'all': 'The lung fields, heart, and surrounding structures appear normal with no significant abnormalities.',
            'unknown': 'The lung fields, heart, and surrounding structures appear normal with no significant abnormalities.'
        },
        'symptoms': 'No symptoms associated with this finding.',
        'causes': 'N/A',
        'implications': 'No further action is needed unless clinical symptoms suggest otherwise.'
    },
    'Nodule': {
        'description': 'A lung nodule is a small, round growth in the lung, typically less than 3 cm in diameter.',
        'location_specific': {
            'left_upper': 'Nodular opacity in the left upper lobe suggestive of a lung nodule.',
            'left_middle': 'Nodular opacity in the left middle lobe suggestive of a lung nodule.',
            'left_lower': 'Nodular opacity in the left lower lobe suggestive of a lung nodule.',
            'right_upper': 'Nodular opacity in the right upper lobe suggestive of a lung nodule.',
            'right_middle': 'Nodular opacity in the right middle lobe suggestive of a lung nodule.',
            'right_lower': 'Nodular opacity in the right lower lobe suggestive of a lung nodule.',
            'central': 'Nodular opacity in the central lung fields suggestive of a lung nodule.',
            'unknown': 'Nodular opacity suggestive of a lung nodule, location unspecified.'
        },
        'symptoms': 'Often asymptomatic, but may cause cough or hemoptysis if large or malignant.',
        'causes': 'Causes include infections, benign tumors, or early-stage lung cancer.',
        'implications': 'Nodules require monitoring or biopsy to assess malignancy risk.'
    },
    'Pleural_Thickening': {
        'description': 'Pleural thickening is the scarring or thickening of the lung lining, often due to chronic inflammation.',
        'location_specific': {
            'left_upper': 'Pleural thickening in the left upper lobe, indicating scarring of the lung lining.',
            'left_middle': 'Pleural thickening in the left middle lobe, indicating scarring of the lung lining.',
            'left_lower': 'Pleural thickening in the left lower lobe, indicating scarring of the lung lining.',
            'right_upper': 'Pleural thickening in the right upper lobe, indicating scarring of the lung lining.',
            'right_middle': 'Pleural thickening in the right middle lobe, indicating scarring of the lung lining.',
            'right_lower': 'Pleural thickening in the right lower lobe, indicating scarring of the lung lining.',
            'central': 'Bilateral pleural thickening, indicating scarring of the lung lining.',
            'unknown': 'Pleural thickening indicating scarring of the lung lining, location unspecified.'
        },
        'symptoms': 'Symptoms may include chest pain and reduced lung capacity, though it can be asymptomatic.',
        'causes': 'Causes include asbestos exposure, infections, or autoimmune diseases.',
        'implications': 'May require monitoring or treatment of the underlying cause.'
    },
    'Pneumonia': {
        'description': 'Pneumonia is an infection that inflames the air sacs in one or both lungs.',
        'location_specific': {
            'left_upper': 'Patchy opacities in the left upper lobe suggestive of pneumonia.',
            'left_middle': 'Patchy opacities in the left middle lobe suggestive of pneumonia.',
            'left_lower': 'Patchy opacities in the left lower lobe suggestive of pneumonia.',
            'right_upper': 'Patchy opacities in the right upper lobe suggestive of pneumonia.',
            'right_middle': 'Patchy opacities in the right middle lobe suggestive of pneumonia.',
            'right_lower': 'Patchy opacities in the right lower lobe suggestive of pneumonia.',
            'central': 'Bilateral patchy opacities suggestive of pneumonia.',
            'unknown': 'Patchy opacities suggestive of pneumonia, location unspecified.'
        },
        'symptoms': 'Symptoms include fever, productive cough, chest pain, and shortness of breath.',
        'causes': 'Caused by bacteria, viruses, or fungi, with bacterial pneumonia being the most common.',
        'implications': 'Requires antibiotics or antiviral treatment, depending on the cause, and supportive care.'
    },
    'Pneumothorax': {
        'description': 'Pneumothorax is a collapsed lung caused by air trapped in the pleural space.',
        'location_specific': {
            'left_upper': 'Absence of lung markings in the left upper lobe suggestive of pneumothorax.',
            'left_middle': 'Absence of lung markings in the left middle lobe suggestive of pneumothorax.',
            'left_lower': 'Absence of lung markings in the left lower lobe suggestive of pneumothorax.',
            'right_upper': 'Absence of lung markings in the right upper lobe suggestive of pneumothorax.',
            'right_middle': 'Absence of lung markings in the right middle lobe suggestive of pneumothorax.',
            'right_lower': 'Absence of lung markings in the right lower lobe suggestive of pneumothorax.',
            'central': 'Bilateral absence of lung markings suggestive of pneumothorax.',
            'unknown': 'Absence of lung markings suggestive of pneumothorax, location unspecified.'
        },
        'symptoms': 'Symptoms include sudden chest pain, shortness of breath, and rapid heart rate.',
        'causes': 'Causes include trauma, lung disease, or spontaneous rupture of air sacs.',
        'implications': 'May require urgent intervention, such as chest tube insertion, to re-expand the lung.'
    }
}


# Transform
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature map analysis
def analyze_feature_maps(features, image_size=(224, 224)):
    try:
        heatmap = torch.mean(features, dim=1).squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, image_size)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        h, w = image_size
        zones = {
            'left_upper': heatmap[:h//3, :w//2],
            'right_upper': heatmap[:h//3, w//2:],
            'central': heatmap[h//4:3*h//4, w//4:3*w//4]
        }
        max_activation = -1
        max_zone = None
        for zone, activation in zones.items():
            avg_activation = np.mean(activation)
            if avg_activation > max_activation:
                max_activation = avg_activation
                max_zone = zone
        return max_zone if max_zone else 'unknown'
    except Exception as e:
        logger.error(f"Failed to analyze feature maps: {e}")
        return 'unknown'

# Helper function to tokenize content while preserving structure
def tokenize_content(content):
    sentences = nltk.sent_tokenize(content)
    return " ".join(sentences)

# Report generation with updated structure while using NLTK
def generate_medical_report(predictions, class_names, image_path, features, threshold=0.2):
    detected_diseases = [
        (class_names[i], prob)
        for i, prob in enumerate(predictions)
        if prob >= threshold and class_names[i] != 'No Finding'
    ]
    no_finding = predictions[class_names.index('No Finding')] >= threshold if 'No Finding' in class_names else False
    location = analyze_feature_maps(features)
    
    # Start directly with Summary of Findings, remove Image ID and Date
    report_lines = [
        "**Summary of Findings**",
        ""  # Ensure new line after subheading
    ]
    
    if no_finding and not detected_diseases:
        desc = DISEASE_DESCRIPTIONS['No Finding']
        report_lines.append(tokenize_content(desc['location_specific']['all']))
    elif detected_diseases:
        report_lines.append("The following conditions were identified with high confidence:")
        report_lines.append("")  # Spacing
        for cls, prob in detected_diseases:
            report_lines.append(f"- {cls}: {prob:.2%}")
    else:
        report_lines.append(tokenize_content("No conditions detected with confidence above the threshold."))
    
    report_lines.append("")  # Spacing
    report_lines.append("**Detailed Analysis**")
    report_lines.append("")  # Ensure new line after subheading
    
    if detected_diseases:
        for idx, (cls, prob) in enumerate(detected_diseases):
            desc = DISEASE_DESCRIPTIONS.get(cls, {})
            location_desc = desc.get('location_specific', {}).get(location, desc.get('description', 'No description available.'))
            report_lines.append(f"**{cls}**")
            report_lines.append("")  # Ensure new line after subheading
            report_lines.append(f"**Radiographic Findings**")
            report_lines.append(tokenize_content(location_desc))
            report_lines.append("")  # Ensure new line after content
            report_lines.append(f"**Clinical Symptoms**")
            report_lines.append(tokenize_content(desc.get('symptoms', 'No clinical symptoms information available.')))
            report_lines.append("")  # Ensure new line after content
            report_lines.append(f"**Differential Diagnosis**")
            report_lines.append(tokenize_content(desc.get('causes', 'No differential diagnosis information available.')))
            report_lines.append("")  # Ensure new line after content
            report_lines.append(f"**Recommendations**")
            report_lines.append(tokenize_content(desc.get('implications', 'No recommendations available.')))
            # Add extra spacing between problems (two newlines)
            if idx < len(detected_diseases) - 1:
                report_lines.append("")
                report_lines.append("")
    
    report_lines.append("")  # Spacing
    report_lines.append("**Interpretation**")
    report_lines.append("")  # Ensure new line after subheading
    report_lines.append(tokenize_content("This report was generated by an AI model. Final interpretation and clinical correlation must be performed by a qualified radiologist."))
    
    report_text = "\n".join(report_lines)
    return report_text

# CAM generator
def get_cam(model, image_tensor, class_idx, device):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.features(image_tensor)
        _ = model.classifier(features.mean([2, 3]))
    weights = model.classifier.weight[class_idx].view(-1, 1, 1)
    cam = torch.sum(features * weights, dim=1).squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, IMAGE_SIZE)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam, features

# Model class
class XrayModel:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, NUM_CLASSES)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open {image_path}: {e}")
            raise
        image_tensor = transform(image).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor.unsqueeze(0))
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        return {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probs)}

    def generate_report_and_cam(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open {image_path}: {e}")
            raise
        image_tensor = transform(image).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor.unsqueeze(0))
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            features = self.model.features(image_tensor.unsqueeze(0))
        top_idx = np.argmax(probs)
        cam, features = get_cam(self.model, image_tensor, top_idx, self.device)
        report = generate_medical_report(probs, CLASS_NAMES, image_path, features, threshold=0.2)
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        overlay = heatmap + img_np
        overlay = overlay / np.max(overlay)
        overlay = (overlay * 255).astype(np.uint8)
        return {
            'predictions': {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probs)},
            'report': report,
            'cam_image': overlay,
            'top_class': CLASS_NAMES[top_idx]
        }


