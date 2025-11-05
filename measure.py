from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
from pathlib import Path
import numpy as np
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
PIXELS_PER_MM = 10

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_a4_paper(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > 40000:
            return approx.reshape(4,2)
    return None

def warp_to_a4(img, pts):
    rect = order_points(pts)
    dst = np.array([
        [0,0],
        [A4_WIDTH_MM*PIXELS_PER_MM-1, 0],
        [A4_WIDTH_MM*PIXELS_PER_MM-1, A4_HEIGHT_MM*PIXELS_PER_MM-1],
        [0, A4_HEIGHT_MM*PIXELS_PER_MM-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (int(A4_WIDTH_MM*PIXELS_PER_MM), int(A4_HEIGHT_MM*PIXELS_PER_MM)))
    return warped

def segment_object_on_a4(warped, debug=False):

    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    

    paper_color = np.median(lab.reshape(-1,3), axis=0)
    
    deltaE = np.sqrt(
        (lab[:,:,1]-paper_color[1])**2 +
        (lab[:,:,2]-paper_color[2])**2
    )
    deltaE = cv2.convertScaleAbs(deltaE)
    
    _, mask = cv2.threshold(deltaE, 10, 255, cv2.THRESH_BINARY)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8), iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No object found")
    
    largest_area = max(cv2.contourArea(c) for c in contours)
    min_area = largest_area * 0.05 
    
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    obj = max(filtered_contours, key=cv2.contourArea)
    
    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [obj], -1, 255, -1)
    
    return obj, final_mask

def measure_object(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None: 
        raise ValueError("Image not found")
    
    a4 = detect_a4_paper(img)
    if a4 is None:
        raise ValueError("A4 sheet not detected")
    
    warped = warp_to_a4(img, a4)
    obj_contour, mask = segment_object_on_a4(warped, debug=debug)
    
    area_px = cv2.contourArea(obj_contour)
    perimeter_px = cv2.arcLength(obj_contour, True)
    
    perimeter_mm = perimeter_px / PIXELS_PER_MM
    perimeter_cm = perimeter_mm / 10

    area_mm2 = area_px / (PIXELS_PER_MM**2)
    area_cm2 = area_mm2 / 100
    
    x,y,w,h = cv2.boundingRect(obj_contour)
    
    vis = warped.copy()
    cv2.drawContours(vis, [obj_contour], -1, (0,255,255), 8)
    
    return round(area_cm2, 2), round(perimeter_cm, 2)

@app.route("/get_area_circumference", methods=["POST"])
def get_area_circumference():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:

        area_cm2, circumference_cm = measure_object(filepath, debug=False)

        vis_filename = f"result_{file.filename}"
        vis_path = os.path.join(RESULT_FOLDER, vis_filename)

        img = cv2.imread(filepath)
        cv2.putText(img, f"A={area_cm2}cm², C={circumference_cm}cm",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imwrite(vis_path, img)

        return jsonify({
            "area_cm2": area_cm2,
            "circumference_cm": circumference_cm,
            "result_image_url": f"/results/{vis_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results/<filename>")
def results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
